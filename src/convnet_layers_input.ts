import { Vol } from "./convnet_vol";
import { LayerBase, LayerOptionsBase, ParamsAndGrads } from "./layers";
import type { ILayer, SerializedLayerBase } from "./layers";
import { getopt } from "./convnet_util";

export interface InputOptions extends LayerOptionsBase<'input'> {
    out_sx: number;
    out_sy: number;
    out_depth: number;
    depth?: number;
    width?: number;
    height?: number;
    sx?: number;
    sy?: number;
}

export interface SerializedInput extends SerializedLayerBase<'input'>{}

export class InputLayer extends LayerBase<'input'> implements ILayer<'input', SerializedInput> {
    out_depth: number;
    out_sx: number;
    out_sy: number;
    in_act: Vol;
    out_act: Vol;

    constructor(opt?: InputOptions) {
        if (!opt) { return; }
        super('input', opt);

        // required: depth
        this.out_depth = getopt(opt, ['out_depth', 'depth'], 0);

        // optional: default these dimensions to 1
        this.out_sx = getopt(opt, ['out_sx', 'sx', 'width'], 1);
        this.out_sy = getopt(opt, ['out_sy', 'sy', 'height'], 1);
    }
    forward(V: Vol, ) {
        this.in_act = V;
        this.out_act = V;
        return this.out_act; // simply identity function for now
    }
    backward() { }
    getParamsAndGrads(): ParamsAndGrads[] {
        return [];
    }
    toJSON(): SerializedInput {
        return {
            layer_type: this.layer_type,
            out_sx: this.out_sx,
            out_sy: this.out_sy,
            out_depth: this.out_depth,
        }
    }
    fromJSON(json: SerializedInput) {
        this.out_depth = json.out_depth as number;
        this.out_sx = json.out_sx as number;
        this.out_sy = json.out_sy as number;
        this.layer_type = json.layer_type as 'input';

        return this
    }
}
