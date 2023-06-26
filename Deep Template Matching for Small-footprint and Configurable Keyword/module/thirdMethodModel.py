import torch


class DeepTemplateMatchingModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        # 1st module
        self.evaluation_layers = torch.nn.ModuleList()
        self.template_layers = torch.nn.ModuleList()

        self.evaluation_layers.append(torch.nn.Conv2d(1, 16, 5))
        self.template_layers.append(torch.nn.Conv2d(1, 16, 5))

        self.evaluation_layers.append(torch.nn.Conv2d(16, 32, 5))
        self.template_layers.append(torch.nn.Conv2d(16, 32, 5))

        self.evaluation_layers.append(torch.nn.Conv2d(32, 64, 5))
        self.template_layers.append(torch.nn.Conv2d(32, 64, 5))

        self.evaluation_layers.append(torch.nn.MaxPool2d((2, 5), (2, 1)))
        self.template_layers.append(torch.nn.MaxPool2d((2, 5), (2, 1)))

        self.evaluation_linear = torch.nn.Linear(3712, 64)
        self.template_linear = torch.nn.Linear(3712, 64)

        self.evaluation_gru = torch.nn.GRU(64, 64)
        self.template_gru = torch.nn.GRU(64, 64)
        # 2nd module
        self.attention = torch.nn.Linear(64, 1)

        # 3rd module
        self.hidden = torch.nn.Linear(64, 128)
        self.activation = torch.nn.ReLU()
        self.classifier = torch.nn.Linear(128, 2)

    def forward(self, evaluation, template):
        evaluation = torch.unsqueeze(evaluation, 1)
        template = torch.unsqueeze(template, 1)

        # feature extraction for evaluation vector
        for layer in self.evaluation_layers:
            evaluation = layer(evaluation)
        if len(evaluation.size()) > 3:
            evaluation = torch.reshape(evaluation, (evaluation.size(0), evaluation.size(-1), 3712))
        else:
            evaluation = torch.reshape(evaluation, (evaluation.size(-1), 3712))
        evaluation = self.evaluation_linear(evaluation)
        evaluation, _ = self.evaluation_gru(evaluation)

        # feature extraction for template vector
        for layer in self.template_layers:
            template = layer(template)
        if len(template.size()) > 3:
            template = torch.reshape(template, (template.size(0), template.size(-1), 3712))
        else:
            template = torch.reshape(template, (template.size(-1), 3712))
        template = self.template_linear(template)
        template, _ = self.template_gru(template)

        # end of 1st module

        # equation (3),(4)
        x = torch.bmm(template, torch.transpose(evaluation, dim0=-2, dim1=-1))

        x = torch.softmax(x, dim=-1)
        
        # equation (5)
        x = torch.bmm(torch.transpose(x, dim0=-2, dim1=-1), template) # x is now T aligned template vectors
        # equation (6)
        x = torch.abs(torch.sub(x, evaluation)) # x is now distance of evaluation and aligned template

        # equation (7), (8)
        attention = torch.softmax(torch.squeeze(self.attention(evaluation)), dim=-1)

        # equation (9)
        temp_stack = []
        for j in range(x.size(-3)):
            one_batch = None
            for i in range(x.size(-2)):
                if one_batch is None:
                    one_batch = torch.mul(x[j, 0], attention[j, 0])
                else:
                    one_batch = torch.add(torch.mul(x[j, i], attention[j, i]), one_batch)
            temp_stack.append(one_batch)
        x = torch.stack(temp_stack) # x.shape is [batch,1,64]

        # end of 2nd module

        x = self.activation(self.hidden(x))
        x = self.classifier(x)

        return torch.softmax(x, -1)
