import { Vol } from "./convnet_vol";
export interface LayerOptionsBase<T extends string> {
    type: T;
    in_sx: number;
    in_sy: number;
    in_depth: number;
}

export interface SerializedLayerBase<T> {
    layer_type: T;
    out_sx: number;
    out_sy: number;
    out_depth: number;
}

export interface ParamsAndGrads {
    params: number[] | Float64Array;
    grads: number[] | Float64Array;
    l2_decay_mul: number;
    l1_decay_mul: number;
}


export interface ILayer<T extends string, S extends SerializedLayerBase<T>> {
    layer_type: T;
    in_sx: number;
    in_sy: number;
    in_depth: number;
    out_sx: number;
    out_sy: number;
    out_depth: number;
    forward(V: Vol, is_training: boolean): Vol;
    backward(y?: number | number[] | Float64Array | { [key: string]: number }): void | number;
    getParamsAndGrads(): ParamsAndGrads[];
    toJSON(): S;
    fromJSON(json: S): ILayer<T, S>;
}

export class LayerBase<T extends string> {
    layer_type: T;
    in_sx: number;
    in_sy: number;
    in_depth: number;
    out_sx: number;
    out_sy: number;
    out_depth: number;
    constructor(layerType: T, opt?: LayerOptionsBase<T>) {
        this.layer_type = layerType;
        if (!opt) { return; }
    }
}