import numpy as np
import torch


class DeepTemplateMatchingModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        # 1st module'
        self.conv1 = torch.nn.Conv2d(1, 16, 5)
        self.conv2 = torch.nn.Conv2d(16, 32, 5)
        self.conv3 = torch.nn.Conv2d(32, 64, 5)
        self.pool = torch.nn.MaxPool2d((5, 2), (1, 2))
        self.linear1 = torch.nn.Linear(3712, 64)
        self.gru = torch.nn.GRU(64, 64)
        # 2nd module
        self.attention = torch.nn.Linear(64, 1)

        # 3rd module
        self.activation = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(64, 128)
        self.classifier = torch.nn.Linear(128, 2)

    def forward(self, data):
        outputs = []

        h0 = None
        h1 = None
        for input_tuple in data:
            evaluation, template = input_tuple
            evaluation = torch.transpose(evaluation, 0, 1)
            template = torch.transpose(template, 0, 1)
            evaluation = torch.reshape(evaluation, (1, evaluation.size(dim=0), 128))
            template = torch.reshape(template, (1, template.size(dim=0), 128))

            evaluation = self.conv1(evaluation)
            template = self.conv1(template)

            evaluation = self.conv2(evaluation)
            template = self.conv2(template)

            evaluation = self.conv3(evaluation)
            template = self.conv3(template)

            evaluation = self.pool(evaluation)
            template = self.pool(template)

            evaluation = torch.reshape(evaluation, (evaluation.size(dim=1), 3712))
            template = torch.reshape(template, (template.size(dim=1), 3712))

            evaluation = self.linear1(evaluation)
            template = self.linear1(template)

            if h1 is None or h0 is None:
                evaluation, h0 = self.gru(evaluation)
                template, h1 = self.gru(template)
            else:
                evaluation, h0 = self.gru(evaluation, h0)
                template, h1 = self.gru(template, h1)

            # print("left the 1st module")

            # print(template.shape)
            alignment_weights = np.zeros((template.size(dim=0), template.size(dim=0)))
            for m in range(template.size(dim=0)):
                for n in range(template.size(dim=0)):
                    alignment_weights[m][n] = torch.dot(evaluation[m, :], template[n, :])
            alignment_weights = torch.from_numpy(alignment_weights)
            alignment_weights = torch.nn.functional.softmax(alignment_weights, dim=0)
            # print(alignment_weights)
            # print(alignment_weights.shape)

            template_prim = np.zeros((template.size(dim=0), (template.size(dim=1))))
            template_prim = torch.from_numpy(template_prim)
            for m in range(template.size(dim=0)):
                for n in range(template.size(dim=0)):
                    template_prim[m] = torch.add(template_prim[m], torch.mul(template[n], alignment_weights[m][n]))

            distance = torch.sub(template_prim, evaluation)
            distance = torch.abs(distance)
            # print(distance.shape)

            attention = self.attention(evaluation)
            attention = torch.flatten(attention)
            attention = torch.nn.functional.softmax(attention, dim=0)
            # print(attention.shape)
            representation = np.zeros(distance.size(dim=1))
            representation = torch.from_numpy(representation)

            # equation 9
            for n in range(distance.size(dim=1)):
                for m in range(distance.size(dim=0)):
                    representation[n] += distance[m][n] * attention[m]

            # print("left the 2nd module")
            output = self.activation(representation)
            output = self.linear3(output.float())
            output = self.activation(output)
            output = self.classifier(output.float())
            outputs.append(output)
        return torch.stack(outputs)
