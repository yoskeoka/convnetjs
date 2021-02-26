// direct export
export { Vol } from "./convnet_vol";
export { Net } from "./convnet_net";
export { MagicNet, MagicNetOptions } from "./convnet_magicnet";
export { randf, randi, randn, randperm } from "./convnet_util";
export { augment, img_to_vol } from './convnet_vol_util';
export { ConvLayer, FullyConnLayer, ConvOptions as ConvLayerOptions, FullyConnOptions as FullyConnLayerOptions } from "./convnet_layers_dotproducts";
export { DropoutLayer } from "./convnet_layers_dropout";
export { RegressionLayer, SVMLayer, SoftmaxLayer, LossLayerOptions } from "./convnet_layers_loss";
export { LocalResponseNormalizationLayer, LocalResponseNormalizationOptions as LocalResponseNormalizationLayerOptions } from "./convnet_layers_normalization";
export { PoolLayer, PoolOptions as PoolLayerOptions } from "./convnet_layers_pool";
export { TrainerOptions } from "./convnet_trainers";

// module export
import * as cnnvis from "./cnnvis";
import * as cnnutil from "./cnnutil";
import * as util from "./convnet_util";
import * as volutil from "./convnet_vol_util";
import * as deepqlearn from "./deepqlearn";

export { cnnvis };
export { cnnutil };
export { util };
export { volutil };
export { deepqlearn };

// rename
import { Trainer } from "./convnet_trainers";
export { Trainer, Trainer as SGDTrainer };

// typings
export type { TrainStats } from "./convnet_trainers";
export type { SerializedNet } from "./convnet_net";
export * from './typings';
