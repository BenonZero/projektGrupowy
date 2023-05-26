from torch import nn, transpose, dot, tensor, softmax, cat, tensor, exp, sum, empty, vstack, zeros


# from math import exp

class SpeechEncDec(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # self.add_module("first_cnn", nn.Conv1d(in_channels=13, out_channels=128, kernel_size=1))
        self.add_module("first_cnn", nn.Conv1d(in_channels=1, out_channels=128, kernel_size=1))
        # "Number of filters = Number of out_channels."
        # "Kernel size of 3 works fine everywhere"
        self.add_module("second_cnn", nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1))
        self.add_module("pool", nn.MaxPool1d(kernel_size=1, stride=2))
        # TODO: more layers in GRUs
        self.add_module("gru_encoder", nn.GRU(input_size=256, hidden_size=256, num_layers=1, batch_first=True))
        self.add_module("encoder_relu", nn.ReLU())
        self.add_module("gru_decoder", nn.GRU(input_size=256, hidden_size=256, num_layers=1, batch_first=True))
        # batched input -> output(L,N,H)
        # 26 is the nb of letters is english alphabet
        # was: 1
        self.add_module("FCL", nn.Linear(in_features=512, out_features=26))

    # def forward(self, X, training : bool):
    def forward(self, X):
        modules = self.named_modules()
        d = dict(modules)

        print(X.size())
        Y = (d["first_cnn"])(X)
        tmp = X
        X = Y
        print(X.size())
        Y = tmp
        Y = (d["second_cnn"])(X)
        tmp = X
        X = Y
        print(X.size())
        Y = tmp

        Y = (d["pool"])(X)
        tmp = X
        X = Y
        print(X.size())
        Y = tmp
        # TODO: wywala przy transpozycji w skrypcie uczacym
        # Y, hidden_encoder = (d["gru_encoder"])(transpose(X, 0, 2))
        # Y, hidden_encoder = (d["gru_encoder"])(X)

        # Y, hidden_encoder = (d["gru_encoder"])(transpose(X, -2, -1))
        # # transposes two last dimensions of X ([...,x,y]->[...,y,x])
        # tmp = X
        # X = Y
        # # X = Y[0] # Y: tuple(data, metadata)
        # print("gru encoder output")
        # print(X.size())
        # print("gru encoder hidden state")
        # print(hidden_encoder.size())
        # Y = tmp

        # Y = (d["encoder_relu"])(X)
        # (X, Y) = (Y, X)
        # print(X.size())

        # if not self.training:
        #     return X # speech embeddings

        # iterowanie po t, wielokrotne puszczanie GRU -> h_t
        # dla kazdego h_t obliczanie attention dla kazdego e_s
        # ze wzoru

        # hidden_encoder = hidden_encoder[-1]

        X = transpose(X, -2, -1)
        predictions_matrix = None

        if len(X.size()) == 2:
            X = X.unsqueeze(1)

        for batch in range(X.size()[0]):
            # first dimension -> batch size

            Y, hidden_encoder = (d["gru_encoder"])(X[batch])
            # hidden_encoder and Y are 2D
            (X, Y) = (Y, X)

            if not predictions_matrix:
                print("encoder out\n", X.size(), "\nencoder h\n", hidden_encoder.size())

            Y = (d["encoder_relu"])(X)
            (X, Y) = (Y, X)

            # h = hidden_encoder[batch]
            h = hidden_encoder
            # h is 2D -> [1,x]
            # embeddings = X
            attention = zeros(X.size()[0])
            # 0 - wymiar s (iterowanie po e_s)
            predictions = None
            for t in range(X.size()[0]):
                # 0 - wymiar t (iterowanie po czasie (sample_rate))
                Y, h = (d["gru_decoder"])(X, h)
                (X, Y) = (Y, X)
                # h_t = h_t[-1]
                # now h_t is a 2D matrix (-1 bc only the last hidden state is of interest)

                for i, e_s in enumerate(X):
                    numerator = exp(dot(h[0], e_s))
                    # h[0] nie wyglada, ale to jest h[t]

                    denominator = None
                    for e_s_prim in X:
                        if denominator is None:
                            denominator = exp(dot(h[0], e_s_prim))
                        else:
                            denominator += exp(dot(h[0], e_s_prim))

                    attention[i] = numerator / denominator

                c_t = X[0] * attention[0]
                # weighted sum of the speech embeddings

                for s, e_s in enumerate(X[1:]):
                    c_t += e_s * attention[s]

                catted = cat((h[0], c_t))
                o_t = (d["FCL"])(catted)
                o_t = softmax(o_t, dim=0)

                if predictions is None:
                    predictions = o_t
                else:
                    predictions = vstack((predictions, o_t))
            if predictions_matrix is None:
                predictions_matrix = predictions
                print("predictions:\n", predictions.size())
            else:
                predictions_matrix = vstack((predictions_matrix, predictions))
        print("predictions_matrix:\n", predictions_matrix.size())
        return predictions_matrix

        # YAGNI

        # Y, hidden = (d["gru_decoder"])(X, hidden_encoder) # transpose?
        # print("gru decoder output")
        # print(Y.size())
        # print("gru decoder hidden state")
        # print(hidden.size())

        # # context = list()

        # # predictions = list()
        # predictions = None

        # # 
        # for h_t in hidden:

        #     attention = empty(0)
        #     for e_s in embeddings:
        #         numerator = exp(dot(h_t, e_s))

        #         denominator = tensor([0.])

        #         for e_s_prim in embeddings:
        #             denominator += exp(dot(h_t, e_s_prim))
        #         attention = cat((attention, (numerator / denominator)))

        #     c_t = embeddings[0] * attention[0]
        #     # weighted sum of the speech embeddings

        #     for s, e_s in enumerate(embeddings[1:]):
        #         c_t += e_s * attention[s]

        #     # context.append(c_t)
        #     catted = cat((h_t, c_t))
        #     # [kitty.item() for kitty in catted]

        #     o_t = (d["FCL"])(catted)
        #     # print("FCL")
        #     # print(o_t)
        #     o_t = softmax(o_t, dim = 0)
        #     # print("softmax")
        #     # print(o_t)
        #     # predictions.append(o_t.item())
        #     # predictions = cat(predictions, o_t)
        #     if predictions is None:
        #         predictions = o_t
        #     else:
        #         predictions = vstack((predictions, o_t))

        # # for h_t in hidden:

        # #     attention = list()
        # #     for e_s in embeddings:
        # #         numerator = exp(dot(h_t, e_s))
        # #         denominator = 0
        # #         for e_s_prim in embeddings:
        # #             denominator += exp(dot(h_t, e_s_prim))
        # #         attention.append(numerator / denominator)
        # #     attention = tensor(attention)

        # #     c_t = embeddings[0] * attention[0]
        # #     # weighted sum of the speech embeddings

        # #     for s, e_s in enumerate(embeddings[1:]):
        # #         c_t += e_s * attention[s]

        # #     # context.append(c_t)
        # #     catted = cat((h_t, c_t))
        # #     # [kitty.item() for kitty in catted]

        # #     o_t = (d["FCL"])(catted)
        # #     # print("FCL")
        # #     # print(o_t)
        # #     o_t = softmax(o_t, dim = 0)
        # #     # print("softmax")
        # #     # print(o_t)
        # #     # predictions.append(o_t.item())
        # #     predictions.append(o_t)

        # # label = ""
        # # print(predictions)
        # # return tensor([0.])
        # # for prediction in predictions:
        # #     local_prediction = list(prediction)
        # #     label += chr(local_prediction.index(max(local_prediction)) + 97)

        # # context = tensor(context)
        # # print(label)
        # # print(len(label))
        # return predictions
