import os
import sys
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from utils.device import set_gpu
from dataloader.dataset import MyDataset
from model.SubNet import get_models
from config.path import RESULTS
from config.tasks import TASKS

sys.path.append(os.getcwd())

EPOCHS = 100  # Reduced for faster training; increase to 1000 for full model
INPUT_DIMS, OUTPUT_DIMS = 5, 13
TASK = 'default'
OUTPUT_FOLDER = 'tmp'

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # CPU:-1; GPU0: 1; GPU1: 0;
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 1: 所有信息, 2: warning & Error, 3: Error

    # --------------------0. set device--------------------
    set_gpu()

    # --------------------1. load data-------------------
    ds = MyDataset(data_file='dataset300.csv').load(task=TASK)
    _, train_x, train_y, _, test_x, test_y = ds.prepare(train_ratio=0.8)

    # --------------------4. build model-----------------
    structures = TASKS[TASK]['models']
    for i in range(len(structures)):
        structures[i].update({
            'input_dims': INPUT_DIMS,
            'output_dims': OUTPUT_DIMS
        })
    models = get_models(structures=structures)
    for i, item in enumerate(models):
        print(f'Training model {item["name"]} ({i + 1}/{len(models)})')
        model = item['model']
        epochs = item['epochs'] if 'epochs' in item.keys() else EPOCHS
        model.compile(loss='mean_squared_error',
                      optimizer=Adam(),
                      metrics=['mse', 'accuracy'])
        # model.load_weights(os.path.join(RESULTS, 'SubNet'))

        # --------------------5. training---------------------
        # 5.1 tensorboard callback
        folder_name = f'{item["name"]}-{epochs}epochs'
        train_result_folder = os.path.join(RESULTS, OUTPUT_FOLDER, folder_name)
        if not os.path.exists(train_result_folder):
            os.makedirs(train_result_folder)
        tb_callback = TensorBoard(log_dir=os.path.join(train_result_folder, 'logs'), histogram_freq=1)

        # 5.2 save model callback
        save_callback = ModelCheckpoint(filepath=os.path.join(train_result_folder, 'SubNet.weights.h5'),
                                        save_weights_only=True,
                                        monitor='loss',
                                        save_best_only=True)

        # 5.3 train with gpu 0
        model.fit(train_x, train_y,
                  batch_size=40,
                  shuffle=True,
                  epochs=epochs,
                  callbacks=[tb_callback, save_callback],
                  validation_data=(test_x, test_y))
