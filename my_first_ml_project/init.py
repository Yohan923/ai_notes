import data_loader
import model
import tensorflow as tf

if __name__ == '__main__':

    data_set = data_loader.load_data()
    train_set = tf.data.Dataset.from_tensor_slices((data_set["train_X"].values, data_set["train_Y"].values))

    train_dataset = train_set.shuffle(len(data_set["train_X"]))

    model = model.Model(train_dataset, train_dataset)
    model.train()

    print("ok")

    """
    predictions = model.predict()

    for prediction, actual in zip(predictions[:10], list(data_set["pred_set"])[0][1][:10]):
        print("Predicted income: {:.2}".format(prediction[0]),
              " | Actual outcome: ",
              actual)
    """
