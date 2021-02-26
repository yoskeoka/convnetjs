import { Vol } from "./convnet_vol";
import { LayerBase, LayerOptionsBase, ParamsAndGrads } from "./layers";
import type { SerializedLayerBase, ILayer } from "./layers";
import * as util from "./convnet_util";

export interface DropoutOptions extends LayerOptionsBase<'dropout'> {
    /** <required> */
    drop_prob: number;
}

export interface SerializedDropout extends SerializedLayerBase<'dropout'> {
    drop_prob: number;
}

/**
 * An inefficient dropout layer
 * Note this is not most efficient implementation since the layer before
 * computed all these activations and now we're just going to drop them :(
 * same goes for backward pass. Also, if we wanted to be efficient at test time
 * we could equivalently be clever and upscale during train and copy pointers during test
 * todo: make more efficient.
 */
export class DropoutLayer extends LayerBase<'dropout'> implements ILayer<'dropout', SerializedDropout> {
    in_act: Vol;
    drop_prob: number;
    dropped: boolean[];
    out_act: Vol;

    constructor(opt?: DropoutOptions) {
        if (!opt) { return; }
        super('dropout', opt);

        // computed
        this.out_sx = opt.in_sx as number;
        this.out_sy = opt.in_sy as number;
        this.out_depth = opt.in_depth as number;
        this.drop_prob = typeof opt.drop_prob !== 'undefined' ? opt.drop_prob : 0.5;
        const d = <number[]>util.zeros(this.out_sx * this.out_sy * this.out_depth);
        this.dropped = d.map((v) => v !== 0);
    }
    forward(V: Vol, is_training?: boolean) {
        this.in_act = V;
        if (typeof (is_training) === 'undefined') { is_training = false; } // default is prediction mode
        const V2 = V.clone();
        const N = V.w.length;
        if (is_training) {
            // do dropout
            for (let i = 0; i < N; i++) {
                if (Math.random() < this.drop_prob) { V2.w[i] = 0; this.dropped[i] = true; } // drop!
                else { this.dropped[i] = false; }
            }
        } else {
            // scale the activations during prediction
            for (let i = 0; i < N; i++) { V2.w[i] *= this.drop_prob; }
        }
        this.out_act = V2;
        return this.out_act; // dummy identity function for now
    }
    backward() {
        const V = this.in_act; // we need to set dw of this
        const chain_grad = this.out_act;
        const n = V.w.length;
        V.dw = util.zeros(n); // zero out gradient wrt data
        for (let i = 0; i < n; i++) {
            if (!(this.dropped[i])) {
                V.dw[i] = chain_grad.dw[i]; // copy over the gradient
            }
        }
    }
    getParamsAndGrads(): ParamsAndGrads[] {
        return [];
    }
    toJSON(): SerializedDropout {
        return {
            layer_type: this.layer_type,
            out_sx: this.out_sx,
            out_sy: this.out_sy,
            out_depth: this.out_depth,
            drop_prob: this.drop_prob,
        }
    }
    fromJSON(json: SerializedDropout) {
        this.out_depth = json.out_depth as number;
        this.out_sx = json.out_sx as number;
        this.out_sy = json.out_sy as number;
        this.layer_type = json.layer_type as 'dropout';
        this.drop_prob = json.drop_prob as number;

        const d = <number[]>util.zeros(this.out_sx * this.out_sy * this.out_depth);
        this.dropped = d.map((v) => v !== 0);

        return this
    }
}
