import { Vol } from "./convnet_vol";
import { LayerBase, LayerOptionsBase, ParamsAndGrads } from "./layers";
import type { ILayer, SerializedLayerBase } from "./layers";
import * as util from "./convnet_util";

export class OutputLayer<T extends string> extends LayerBase<T> {
    in_act: Vol;
    out_act: Vol;
    constructor(layerType: T, opt: LayerOptionsBase<T>) {
        super(layerType, opt);
        this.out_sx = opt.in_sx as number;
        this.out_sy = opt.in_sy as number;
        this.out_depth = opt.in_depth as number;
    }
}

export interface ReluOptions extends LayerOptionsBase<'relu'> {}
export interface SerializedRelu extends SerializedLayerBase<'relu'>{}

/**
 * Implements ReLU nonlinearity elementwise
 * x -> max(0, x)
 * the output is in [0, inf)
 */
export class ReluLayer extends OutputLayer<'relu'> implements ILayer<'relu', SerializedRelu> {

    constructor(opt?: ReluOptions) {
        if (!opt) { return; }
        super('relu', opt);
    }
    forward(V: Vol, ) {
        this.in_act = V;
        const V2 = V.clone();
        const N = V.w.length;
        const V2w = V2.w;
        for (let i = 0; i < N; i++) {
            if (V2w[i] < 0) {
                V2w[i] = 0; // threshold at 0
            }
            this.out_act = V2;
            return this.out_act;
        }
    }

    backward() {
        const V = this.in_act; // we need to set dw of this
        const V2 = this.out_act;
        const N = V.w.length;
        V.dw = util.zeros(N); // zero out gradient wrt data
        for (let i = 0; i < N; i++) {
            if (V2.w[i] <= 0) {
                V.dw[i] = 0; // threshold
            }
            else {
                V.dw[i] = V2.dw[i];
            }
        }
    }
    getParamsAndGrads(): ParamsAndGrads[] {
        return [];
    }
    toJSON(): SerializedRelu {
        return {
            layer_type: this.layer_type,
            out_sx: this.out_sx,
            out_sy: this.out_sy,
            out_depth: this.out_depth,
        }
    }
    fromJSON(json: SerializedRelu) {
        this.out_depth = json.out_depth as number;
        this.out_sx = json.out_sx as number;
        this.out_sy = json.out_sy as number;
        this.layer_type = json.layer_type as 'relu';

        return this
    }
}

export interface SigmoidOptions extends LayerOptionsBase<'sigmoid'> {}
export interface SerializedSigmoid extends SerializedLayerBase<'sigmoid'>{}

/**
 * Implements Sigmoid nnonlinearity elementwise
 * x -> 1/(1+e^(-x))
 * so the output is between 0 and 1.
 */
export class SigmoidLayer extends OutputLayer<'sigmoid'> implements ILayer<'sigmoid', SerializedSigmoid> {

    constructor(opt?: SigmoidOptions) {
        if (!opt) { return; }
        super('sigmoid', opt);

        // computed
        this.layer_type = 'sigmoid';
    }
    forward(V: Vol, ) {
        this.in_act = V;
        const V2 = V.cloneAndZero();
        const N = V.w.length;
        const V2w = V2.w;
        const Vw = V.w;
        for (let i = 0; i < N; i++) {
            V2w[i] = 1.0 / (1.0 + Math.exp(-Vw[i]));
        }
        this.out_act = V2;
        return this.out_act;
    }
    backward() {
        const V = this.in_act; // we need to set dw of this
        const V2 = this.out_act;
        const N = V.w.length;
        V.dw = util.zeros(N); // zero out gradient wrt data
        for (let i = 0; i < N; i++) {
            const v2wi = V2.w[i];
            V.dw[i] = v2wi * (1.0 - v2wi) * V2.dw[i];
        }
    }
    getParamsAndGrads(): ParamsAndGrads[] {
        return [];
    }
    toJSON(): SerializedSigmoid {
        return {
            layer_type: this.layer_type,
            out_sx: this.out_sx,
            out_sy: this.out_sy,
            out_depth: this.out_depth,
        }
    }
    fromJSON(json: SerializedSigmoid) {
        this.out_depth = json.out_depth as number;
        this.out_sx = json.out_sx as number;
        this.out_sy = json.out_sy as number;
        this.layer_type = json.layer_type as 'sigmoid';

        return this
    }
}

export interface MaxoutOptions extends LayerOptionsBase<'maxout'> {
    /** <required> group_size must be the integral multiple of the input size   */
    group_size: number;
}
export interface SerializedMaxout extends SerializedLayerBase<'maxout'>{
    group_size: number;
}

// Implements Maxout nnonlinearity that computes
// x -> max(x)
// where x is a vector of size group_size. Ideally of course,
// the input size should be exactly divisible by group_size
export class MaxoutLayer extends OutputLayer<'maxout'> implements ILayer<'maxout', SerializedMaxout> {
    group_size: number;
    switches: number[] | Float64Array;


    constructor(opt?: MaxoutOptions) {
        if (!opt) { return; }
        super('maxout', opt);

        // required
        this.group_size = typeof opt.group_size !== 'undefined' ? opt.group_size : 2;

        // computed
        this.out_depth = Math.floor(<number>opt.in_depth / this.group_size);

        this.switches = util.zeros(this.out_sx * this.out_sy * this.out_depth); // useful for backprop
    }
    forward(V: Vol, ) {
        this.in_act = V;
        const N = this.out_depth;
        const V2 = new Vol(this.out_sx, this.out_sy, this.out_depth, 0.0);

        // optimization branch. If we're operating on 1D arrays we dont have
        // to worry about keeping track of x,y,d coordinates inside
        // input volumes. In convnets we do :(
        if (this.out_sx === 1 && this.out_sy === 1) {
            for (let i = 0; i < N; i++) {
                const ix = i * this.group_size; // base index offset
                let a = V.w[ix];
                let ai = 0;
                for (let j = 1; j < this.group_size; j++) {
                    const a2 = V.w[ix + j];
                    if (a2 > a) {
                        a = a2;
                        ai = j;
                    }
                }
                V2.w[i] = a;
                this.switches[i] = ix + ai;
            }
        } else {
            let n = 0; // counter for switches
            for (let x = 0; x < V.sx; x++) {
                for (let y = 0; y < V.sy; y++) {
                    for (let i = 0; i < N; i++) {
                        const ix = i * this.group_size;
                        let a = V.get(x, y, ix);
                        let ai = 0;
                        for (let j = 1; j < this.group_size; j++) {
                            const a2 = V.get(x, y, ix + j);
                            if (a2 > a) {
                                a = a2;
                                ai = j;
                            }
                        }
                        V2.set(x, y, i, a);
                        this.switches[n] = ix + ai;
                        n++;
                    }
                }
            }

        }
        this.out_act = V2;
        return this.out_act;
    }
    backward() {
        const V = this.in_act; // we need to set dw of this
        const V2 = this.out_act;
        const N = this.out_depth;
        V.dw = util.zeros(V.w.length); // zero out gradient wrt data

        // pass the gradient through the appropriate switch
        if (this.out_sx === 1 && this.out_sy === 1) {
            for (let i = 0; i < N; i++) {
                const chain_grad = V2.dw[i];
                V.dw[this.switches[i]] = chain_grad;
            }
        } else {
            // bleh okay, lets do this the hard way
            let n = 0; // counter for switches
            for (let x = 0; x < V2.sx; x++) {
                for (let y = 0; y < V2.sy; y++) {
                    for (let i = 0; i < N; i++) {
                        const chain_grad = V2.get_grad(x, y, i);
                        V.set_grad(x, y, this.switches[n], chain_grad);
                        n++;
                    }
                }
            }
        }
    }
    getParamsAndGrads(): ParamsAndGrads[] {
        return [];
    }
    toJSON(): SerializedMaxout {
        return {
            layer_type: this.layer_type,
            out_sx: this.out_sx,
            out_sy: this.out_sy,
            out_depth: this.out_depth,
            group_size: this.group_size,
        }
    }
    fromJSON(json: SerializedMaxout) {
        this.out_depth = json.out_depth as number;
        this.out_sx = json.out_sx as number;
        this.out_sy = json.out_sy as number;
        this.layer_type = json.layer_type as 'maxout';
        this.group_size = json.group_size as number;
        this.switches = util.zeros(this.group_size);

        return this
    }
}

export interface TanhOptions extends LayerOptionsBase<'tanh'> {}
export interface SerializedTanh extends SerializedLayerBase<'tanh'>{}

/**
 * a helper function, since tanh is not yet part of ECMAScript. Will be in v6.
 */
function tanh(x: number) {
    const y = Math.exp(2 * x);
    return (y - 1) / (y + 1);
}
// Implements Tanh nnonlinearity elementwise
// x -> tanh(x)
// so the output is between -1 and 1.
export class TanhLayer extends OutputLayer<'tanh'> implements ILayer<'tanh', SerializedTanh> {
    constructor(opt?: TanhOptions) {
        if (!opt) { return; }
        super('tanh', opt);

        // computed
        this.layer_type = 'tanh';
    }
    forward(V: Vol, ) {
        this.in_act = V;
        const V2 = V.cloneAndZero();
        const N = V.w.length;
        for (let i = 0; i < N; i++) {
            V2.w[i] = tanh(V.w[i]);
        }
        this.out_act = V2;
        return this.out_act;
    }
    backward() {
        const V = this.in_act; // we need to set dw of this
        const V2 = this.out_act;
        const N = V.w.length;
        V.dw = util.zeros(N); // zero out gradient wrt data
        for (let i = 0; i < N; i++) {
            const v2wi = V2.w[i];
            V.dw[i] = (1.0 - v2wi * v2wi) * V2.dw[i];
        }
    }
    getParamsAndGrads(): ParamsAndGrads[] {
        return [];
    }
    toJSON(): SerializedTanh {
        return {
            layer_type: this.layer_type,
            out_sx: this.out_sx,
            out_sy: this.out_sy,
            out_depth: this.out_depth,
        }
    }
    fromJSON(json: SerializedTanh) {
        this.out_depth = json.out_depth as number;
        this.out_sx = json.out_sx as number;
        this.out_sy = json.out_sy as number;
        this.layer_type = json.layer_type as 'tanh';

        return this
    }
}
