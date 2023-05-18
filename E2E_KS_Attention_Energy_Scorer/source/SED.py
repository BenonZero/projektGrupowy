from torch import nn, transpose, dot, tensor, softmax, cat, tensor, exp, sum, empty, vstack
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
        self.add_module("gru_encoder", nn.GRU(input_size = 256, hidden_size = 256, num_layers = 256))
        self.add_module("encoder_relu", nn.ReLU())
        self.add_module("gru_decoder", nn.GRU(input_size = 256, hidden_size = 256, num_layers = 256))
        # batched input -> output(L,N,H)
        # 26 is the nb of letters is english alphabet
        # was: 1
        self.add_module("FCL", nn.Linear(in_features = 512, out_features = 26))

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
        #TODO: wywala przy transpozycji w skrypcie uczacym
        Y = (d["gru_encoder"])(transpose(X, 0, 2))
        # Y = (d["gru_encoder"])(X)
        tmp = X
        X = Y
        # X = Y[0] # Y: tuple(data, metadata)
        print(X.size())
        Y = tmp

        Y = (d["encoder_relu"])(X)
        tmp = X
        X = Y
        print(X.size())
        Y = tmp

        if not self.training:
            return X # speech embeddings
        
        # hidden = (d["gru_decoder"])(X)[0] # transpose?
        hidden = (d["gru_decoder"])(X) # transpose?
        print(hidden.size())

        embeddings = X
        # context = list()

        # predictions = list()
        predictions = None

        # 
        for h_t in hidden:
            
            attention = empty(0)
            for e_s in embeddings:
                numerator = exp(dot(h_t, e_s))

                denominator = tensor([0.])
                
                for e_s_prim in embeddings:
                    denominator += exp(dot(h_t, e_s_prim))
                attention = cat((attention, (numerator / denominator)))

            c_t = embeddings[0] * attention[0]
            # weighted sum of the speech embeddings

            for s, e_s in enumerate(embeddings[1:]):
                c_t += e_s * attention[s]

            # context.append(c_t)
            catted = cat((h_t, c_t))
            # [kitty.item() for kitty in catted]

            o_t = (d["FCL"])(catted)
            # print("FCL")
            # print(o_t)
            o_t = softmax(o_t, dim = 0)
            # print("softmax")
            # print(o_t)
            # predictions.append(o_t.item())
            # predictions = cat(predictions, o_t)
            if predictions is None:
                predictions = o_t
            else:
                predictions = vstack((predictions, o_t))








        # for h_t in hidden:
            
        #     attention = list()
        #     for e_s in embeddings:
        #         numerator = exp(dot(h_t, e_s))
        #         denominator = 0
        #         for e_s_prim in embeddings:
        #             denominator += exp(dot(h_t, e_s_prim))
        #         attention.append(numerator / denominator)
        #     attention = tensor(attention)

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
        #     predictions.append(o_t)

        label = ""
        # print(predictions)
        # return tensor([0.])
        # for prediction in predictions:
        #     local_prediction = list(prediction)
        #     label += chr(local_prediction.index(max(local_prediction)) + 97)

        # context = tensor(context)
        # print(label)
        # print(len(label))
        return predictions