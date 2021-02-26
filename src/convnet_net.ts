import { Vol } from "./convnet_vol";
import { ParamsAndGrads } from "./layers";
import * as util from "./convnet_util";

import { SVMLayer, RegressionLayer, SoftmaxLayer } from "./convnet_layers_loss";
import { FullyConnLayer, ConvLayer } from "./convnet_layers_dotproducts";
import { MaxoutLayer, TanhLayer, SigmoidLayer, ReluLayer } from "./convnet_layers_nonlinearities";
import { PoolLayer } from "./convnet_layers_pool";
import { InputLayer } from "./convnet_layers_input";
import { DropoutLayer } from "./convnet_layers_dropout";
import { LocalResponseNormalizationLayer } from "./convnet_layers_normalization";

import { SerializedLayerType, LayerType,  LayerOptionsType, LayerOptionsSugarType } from "./typings";

const assert = util.assert;

export interface SerializedNet {
    layers?: SerializedLayerType[];
}

class InvalidCostTypeError extends TypeError {
    name = 'InvalidCostType'
    constructor() { 
        super('Invalid cost type')
    }
}

export const smartBackward = (y: number | number[] | Float64Array | { [key: string]: number }, layer: LayerType): number => {
    if (layer instanceof RegressionLayer) {
        return layer.backward(y);
    } else {
        if (y instanceof Object) {
            throw new InvalidCostTypeError();
        }

        return layer.backward(y) || 0;
    }
}

/**
 * Net manages a set of layers
 * For now constraints: Simple linear order of layers, first layer input last layer a cost layer
 */
export class Net {
    layers: LayerType[];
    constructor(options?: LayerOptionsType[]) {
        if(!options){
            options = [];
        }
        this.layers = [];
    }
    // takes a list of layer definitions and creates the network layer objects
    makeLayers(defs: LayerOptionsSugarType[]) {

        // few checks
        assert(defs.length >= 2, 'Error! At least one input layer and one loss layer are required.');
        assert(defs[0].type === 'input', 'Error! First layer must be the input layer, to declare size of inputs');

        // desugar layer_defs for adding activation, dropout layers etc
        const desugar = function (defs: LayerOptionsSugarType[]) {
            const new_defs: LayerOptionsSugarType[] = [];
            for (let i = 0; i < defs.length; i++) {
                const def = defs[i];

                if (def.type === 'softmax' || def.type === 'svm' || def.type === 'regression') {
                    // add an fc layer here, there is no reason the user should
                    // have to worry about this and we almost always want to
                    new_defs.push({ type: 'fc', num_neurons: def.num_classes });
                }

                if (def.type === 'fc' || def.type === 'conv') {
                    if (typeof (def.bias_pref) === 'undefined') {
                        def.bias_pref = 0.0;
                        if (typeof def.activation !== 'undefined' && def.activation === 'relu') {
                            def.bias_pref = 0.1; // relus like a bit of positive bias to get gradients early
                            // otherwise it's technically possible that a relu unit will never turn on (by chance)
                            // and will never get any gradient and never contribute any computation. Dead relu.
                        }
                    }
                }

                new_defs.push(def);

                if ("activation" in def) {
                    if (def.activation === 'relu') { new_defs.push({ type: 'relu' }); }
                    else if (def.activation === 'sigmoid') { new_defs.push({ type: 'sigmoid' }); }
                    else if (def.activation === 'tanh') { new_defs.push({ type: 'tanh' }); }
                    else if (def.activation === 'maxout') {
                        // create maxout activation, and pass along group size, if provided
                        const gs = 'group_size' in def ? def.group_size : 2;
                        new_defs.push({ type: 'maxout', group_size: gs });
                    }
                    else { console.log('ERROR unsupported activation ' + def.activation); }
                }
                if ('drop_prob' in def && def.type !== 'dropout') {
                    new_defs.push({ type: 'dropout', drop_prob: def.drop_prob });
                }

            }
            return new_defs;
        }
        defs = desugar(defs);

        // create the layers
        this.layers = [];
        for (let i = 0; i < defs.length; i++) {
            const def = defs[i] as LayerOptionsType;
            if (i > 0) {
                const prev = this.layers[i - 1];
                def.in_sx = prev.out_sx;
                def.in_sy = prev.out_sy;
                def.in_depth = prev.out_depth;
            }

            switch (def.type) {
                case 'fc': this.layers.push(new FullyConnLayer(def)); break;
                case 'lrn': this.layers.push(new LocalResponseNormalizationLayer(def)); break;
                case 'dropout': this.layers.push(new DropoutLayer(def)); break;
                case 'input': this.layers.push(new InputLayer(def)); break;
                case 'softmax': this.layers.push(new SoftmaxLayer(def)); break;
                case 'regression': this.layers.push(new RegressionLayer(def)); break;
                case 'conv': this.layers.push(new ConvLayer(def)); break;
                case 'pool': this.layers.push(new PoolLayer(def)); break;
                case 'relu': this.layers.push(new ReluLayer(def)); break;
                case 'sigmoid': this.layers.push(new SigmoidLayer(def)); break;
                case 'tanh': this.layers.push(new TanhLayer(def)); break;
                case 'maxout': this.layers.push(new MaxoutLayer(def)); break;
                case 'svm': this.layers.push(new SVMLayer(def)); break;
            }
        }
    }

    // forward prop the network.
    // The trainer class passes is_training = true, but when this function is
    // called from outside (not from the trainer), it defaults to prediction mode
    forward(V: Vol, is_training?: boolean) {
        if (typeof (is_training) === 'undefined') { is_training = false; }
        let act = this.layers[0].forward(V, is_training);
        for (let i = 1; i < this.layers.length; i++) {
            act = this.layers[i].forward(act, is_training);
        }
        return act;
    }

    getCostLoss(V: Vol, y: number | number[] | Float64Array | { [key: string]: number }): number {
        this.forward(V, false);
        const N = this.layers.length;

        return smartBackward(y, this.layers[N - 1]);
    }

    /**
     * backprop: compute gradients wrt all parameters
     */
    backward(y: number | number[] | Float64Array | { [key: string]: number }): number {
        const N = this.layers.length;
        const loss = smartBackward(y, this.layers[N - 1]); // last layer assumed to be loss layer
        for (let i = N - 2; i >= 0; i--) { // first layer assumed input
            this.layers[i].backward();
        }
        return loss;
    }
    getParamsAndGrads(): ParamsAndGrads[] {
        // accumulate parameters and gradients for the entire network
        const response = [];
        for (let i = 0; i < this.layers.length; i++) {
            const layer_reponse = this.layers[i].getParamsAndGrads();
            for (let j = 0; j < layer_reponse.length; j++) {
                response.push(layer_reponse[j]);
            }
        }
        return response;
    }
    getPrediction() {
        // this is a convenience function for returning the argmax
        // prediction, assuming the last layer of the net is a softmax
        const S = this.layers[this.layers.length - 1];
        assert(S.layer_type === 'softmax', 'getPrediction function assumes softmax as last layer of the net!');
        if (S instanceof SoftmaxLayer) {
            const p = S.out_act.w;
            let maxv = p[0];
            let maxi = 0;
            for (let i = 1; i < p.length; i++) {
                if (p[i] > maxv) { maxv = p[i]; maxi = i; }
            }
            return maxi; // return index of the class with highest class probability
        }
        throw Error("to getPrediction, the last layer must be softmax");
    }
    toJSON(): SerializedNet {
        const json: SerializedNet = {};
        json.layers = [];
        for (let i = 0; i < this.layers.length; i++) {
            json.layers.push(this.layers[i].toJSON());
        }
        return json;
    }
    fromJSON(json: SerializedNet) {
        this.layers = [];
        for (let i = 0; i < json.layers.length; i++) {
            const layer = json.layers[i];
            let L: LayerType;

            if (layer.layer_type === 'input') { L = new InputLayer().fromJSON(layer); }
            if (layer.layer_type === 'relu') { L = new ReluLayer().fromJSON(layer); }
            if (layer.layer_type === 'sigmoid') { L = new SigmoidLayer().fromJSON(layer); }
            if (layer.layer_type === 'tanh') { L = new TanhLayer().fromJSON(layer); }
            if (layer.layer_type === 'dropout') { L = new DropoutLayer().fromJSON(layer); }
            if (layer.layer_type === 'conv') { L = new ConvLayer().fromJSON(layer); }
            if (layer.layer_type === 'pool') { L = new PoolLayer().fromJSON(layer); }
            if (layer.layer_type === 'lrn') { L = new LocalResponseNormalizationLayer().fromJSON(layer); }
            if (layer.layer_type === 'softmax') { L = new SoftmaxLayer().fromJSON(layer); }
            if (layer.layer_type === 'regression') { L = new RegressionLayer().fromJSON(layer); }
            if (layer.layer_type === 'fc') { L = new FullyConnLayer().fromJSON(layer); }
            if (layer.layer_type === 'maxout') { L = new MaxoutLayer().fromJSON(layer); }
            if (layer.layer_type === 'svm') { L = new SVMLayer().fromJSON(layer); }

            this.layers.push(L);
        }
    }
}
