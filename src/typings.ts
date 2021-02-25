import { SVMLayer, RegressionLayer, SoftmaxLayer } from "./convnet_layers_loss";
import { FullyConnLayer, ConvLayer } from "./convnet_layers_dotproducts";
import { MaxoutLayer, TanhLayer, SigmoidLayer, ReluLayer } from "./convnet_layers_nonlinearities";
import { PoolLayer } from "./convnet_layers_pool";
import { InputLayer } from "./convnet_layers_input";
import { DropoutLayer } from "./convnet_layers_dropout";
import { LocalResponseNormalizationLayer } from "./convnet_layers_normalization";

import type { SerializedSVM, SerializedRegression, SerializedSoftmax } from "./convnet_layers_loss";
import type { SerializedFullyConn, SerializedConv } from "./convnet_layers_dotproducts";
import type { SerializedMaxout, SerializedTanh, SerializedSigmoid, SerializedRelu } from "./convnet_layers_nonlinearities";
import type { SerializedPool } from "./convnet_layers_pool";
import type { SerializedInput } from "./convnet_layers_input";
import type { SerializedDropout } from "./convnet_layers_dropout";
import type { SerializedLocalResponseNormalization } from "./convnet_layers_normalization" ;

import type { LayerOptionsBase } from "./layers";
import type { SVMOptions, RegressionOptions, SoftmaxOptions } from "./convnet_layers_loss";
import type { FullyConnOptions, ConvOptions } from "./convnet_layers_dotproducts";
import type { MaxoutOptions, TanhOptions, SigmoidOptions, ReluOptions } from "./convnet_layers_nonlinearities";
import type { PoolOptions } from "./convnet_layers_pool";
import type { InputOptions } from "./convnet_layers_input";
import type { DropoutOptions } from "./convnet_layers_dropout";
import type { LocalResponseNormalizationOptions } from "./convnet_layers_normalization" ;

export type SerializedLayerType = 
    | SerializedSVM
    | SerializedRegression
    | SerializedSoftmax
    | SerializedFullyConn
    | SerializedConv
    | SerializedMaxout
    | SerializedTanh
    | SerializedSigmoid
    | SerializedRelu
    | SerializedPool
    | SerializedInput
    | SerializedDropout
    | SerializedLocalResponseNormalization;

export type LayerType = 
    | FullyConnLayer
    | LocalResponseNormalizationLayer
    | DropoutLayer
    | InputLayer
    | SoftmaxLayer
    | RegressionLayer
    | ConvLayer
    | PoolLayer
    | ReluLayer
    | SigmoidLayer
    | TanhLayer
    | MaxoutLayer
    | SVMLayer;

export type LayerOptionsType = 
    | FullyConnOptions
    | LocalResponseNormalizationOptions
    | DropoutOptions
    | InputOptions
    | SoftmaxOptions
    | RegressionOptions
    | ConvOptions
    | PoolOptions
    | ReluOptions
    | SigmoidOptions
    | TanhOptions
    | MaxoutOptions
    | SVMOptions;

type Sugar<P extends LayerOptionsBase<any>> = Omit<P, 'in_sx' | 'in_sy' | 'in_depth'>;

export type LayerOptionsSugarType = (
    | Sugar<FullyConnOptions>
    | Sugar<LocalResponseNormalizationOptions>
    | Sugar<DropoutOptions>
    | Sugar<InputOptions>
    | Sugar<SoftmaxOptions>
    | Sugar<RegressionOptions>
    | Sugar<ConvOptions>
    | Sugar<PoolOptions>
    | Sugar<ReluOptions>
    | Sugar<SigmoidOptions>
    | Sugar<TanhOptions>
    | Sugar<MaxoutOptions>
    | Sugar<SVMOptions>
) & {
    activation?: 'relu' | 'tanh' | 'sigmoid' | 'maxout';
    group_size?: number;
};