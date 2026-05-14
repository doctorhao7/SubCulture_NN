import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import Sequential


def get_model(input_dims, output_dims):
    model = Sequential()
    model.add(Dense(units=32, activation='relu', input_shape=(input_dims,), dtype='float64'))
    model.add(Dropout(rate=0.2, dtype='float64'))
    model.add(Dense(units=32, activation='relu', dtype='float64'))
    model.add(Dropout(rate=0.2, dtype='float64'))
    # model.add(Dense(units=32, activation='relu', dtype='float64'))
    # model.add(Dropout(rate=0.2, dtype='float64'))
    model.add(Dense(units=output_dims, activation=None, dtype='float64'))
    return model


def get_models(structures):
    models = []
    is_input_layer = True
    for structure in structures:
        model = Sequential()
        name = ''
        for layer_units in structure['layers']:
            if is_input_layer:
                layer = Dense(units=layer_units, activation='relu', input_shape=(structure['input_dims'],), dtype='float64')
            else:
                layer = Dense(units=layer_units, activation='relu')
            model.add(layer)
            if 'dropout' in structure.keys():
                model.add(Dropout(rate=structure['dropout'], dtype='float64'))
            name += f'{layer_units}-'
        model.add(Dense(units=structure['output_dims'], activation=None, dtype='float64'))
        name = name[:-1]
        item = {
            'name': name,
            'model': model
        }
        if 'epochs' in structure.keys():
            item['epochs'] = structure['epochs']
        models.append(item)
    return models
