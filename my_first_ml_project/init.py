import data_loader
import model

if __name__ == '__main__':

    train_data = data_loader.load_data("data/tcd ml 2019-20 income prediction training (with labels).csv", "train")
    test_data = data_loader.load_data("data/tcd ml 2019-20 income prediction test (without labels).csv", "test")
    model = model.Model(train_data, test_data)
    model.train()

    predictions = model.predict()

    for prediction, actual in zip(predictions[:10], list(test_data)[0][1][:10]):
        print("Predicted income: {:.2}".format(prediction[0]),
              " | Actual outcome: ",
              actual)
