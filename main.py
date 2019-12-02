import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils import clip_grad_norm_
from data_utils import Dictionary, Corpus

# [COMENTARIO]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

embed_size = 128        # [COMENTARIO]
hidden_size = 1024      # [COMENTARIO]
num_layers = 1          # [COMENTARIO]
num_epochs = 5          # [COMENTARIO]
num_samples = 1000      # [COMENTARIO]
batch_size = 20         # [COMENTARIO]
seq_length = 30         # [COMENTARIO]
learning_rate = 0.002   # [COMENTARIO]

# [COMENTARIO]
corpus = Corpus()
ids = corpus.get_data('data/train.txt', batch_size)
vocab_size = len(corpus.dictionary)
num_batches = ids.size(1) // seq_length


class RNNLM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(RNNLM, self).__init__()
        # [COMENTARIO]
        self.embed = nn.Embedding(vocab_size, embed_size)
        # [COMENTARIO]
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        # [COMENTARIO]
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x, h):
        # [COMENTARIO]
        x = self.embed(x)
        
        # [COMENTARIO]
        out, (h, c) = self.lstm(x, h)
        
        # [COMENTARIO]
        out = out.reshape(out.size(0)*out.size(1), out.size(2))
        
        # [COMENTARIO]
        out = self.linear(out)
        return out, (h, c)

# [COMENTARIO] 
model = RNNLM(vocab_size, embed_size, hidden_size, num_layers).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# [COMENTARIO] Pista: https://stats.stackexchange.com/questions/172035/truncated-back-propagation-through-time-for-rnns
def detach(states):
    return [state.detach() for state in states] 

# [COMENTARIO] 
for epoch in range(num_epochs):
    # [COMENTARIO] 
    states = (torch.zeros(num_layers, batch_size, hidden_size).to(device),
              torch.zeros(num_layers, batch_size, hidden_size).to(device))
    
    for i in range(0, ids.size(1) - seq_length, seq_length):
        # [COMENTARIO] 
        inputs = ids[:, i:i+seq_length].to(device)
        targets = ids[:, (i+1):(i+1)+seq_length].to(device)
        
        # [COMENTARIO] 
        states = detach(states)
        outputs, states = model(inputs, states)
        loss = criterion(outputs, targets.reshape(-1))
        
        model.zero_grad()
        loss.backward()

        # [COMENTARIO] ?Por qu\'e usamos esto? 
        clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        step = (i+1) // seq_length
        if step % 100 == 0:
            print ('Epoch [{}/{}], Step[{}/{}], Loss: {:.4f}, Perplexity: {:5.2f}'
                   .format(epoch+1, num_epochs, step, num_batches, loss.item(), np.exp(loss.item())))

# Prueba
# [COMENTARIO]
with torch.no_grad():
    with open('sample.txt', 'w') as f:
        # [COMENTARIO]
        state = (torch.zeros(num_layers, 1, hidden_size).to(device),
                 torch.zeros(num_layers, 1, hidden_size).to(device))

        # [COMENTARIO]
        prob = torch.ones(vocab_size)
        input = torch.multinomial(prob, num_samples=1).unsqueeze(1).to(device)

        for i in range(num_samples):
            # [COMENTARIO]
            output, state = model(input, state)

            # [COMENTARIO]
            prob = output.exp()
            word_id = torch.multinomial(prob, num_samples=1).item()

            # [COMENTARIO]
            input.fill_(word_id)

            # [COMENTARIO]
            word = corpus.dictionary.idx2word[word_id]
            word = '\n' if word == '<eos>' else word + ' '
            f.write(word)

            if (i+1) % 100 == 0:
                print('Sampled [{}/{}] words and save to {}'.format(i+1, num_samples, 'sample.txt'))
