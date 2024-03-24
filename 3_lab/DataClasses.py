import pandas
from pathlib import Path
from torch.utils.data import Dataset
import torch
from torch.nn import Embedding
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import numpy as np
import Baseline

def addWord(dictionary: dict, word: str):
    if word in dictionary:
        dictionary[word]+=1
    else:
        dictionary[word] = 1

def frequencyDict(data: []):
    i=0
    freqDict=dict()
    for row in data:
        words=row.strip().split(" ")
        for word in words:
            addWord(freqDict, word)
    return freqDict

class Instance:
    text = str
    label = str

    def __init__(self, text: str, label: str):
        self.text = text
        self.label = label

class Vocab:
    def __init__(self, frequencies: dict, maxSize: int = -1, minFreq: int = 0, isTextArray: bool = True):
        sortedDict=dict(sorted(frequencies.items(), key=lambda item: item[1], reverse = True))
        self.maxSize = maxSize
        self.minFreq = minFreq
        self.isTextArray = isTextArray
        self.stoi = dict()
        self.itos = dict()
        if isTextArray:
            self.stoi["<PAD>"] = 0
            self.stoi["<UNK>"] = 1
            i=2
        else:
            i=0
        for key in sortedDict.keys():
            if(self.maxSize>=0 and i>=maxSize):
                break
            if sortedDict[key] >= minFreq:
                self.stoi[key] = i
                self.itos[i] = key
                i+= 1

    def encode(self, text: []):
        output=[]
        for word in text:
            if word in self.stoi:
                output.append(self.stoi[word])
            else:
                output.append(self.stoi["<UNK>"])
        return torch.tensor(output)

class NLPDataset(Dataset):
    def __init__(self, textVocabulary: Vocab, labelVocabulary: Vocab, data: []):
        self.textVocab = textVocabulary
        self.labelVocab = labelVocabulary
        self.data = data

    def __getitem__(self, idx: str):
        text = self.data[0][idx].replace("\r\n","").split(" ")
        label = self.data[1][idx].strip()
        return (self.textVocab.encode(text), self.labelVocab.encode([label]))

    def __len__(self):
        return len(self.data)

def generateEmbeddingMatrix(vocabulary: Vocab, path: Path, dimension: int = 300,  freeze: bool = True):
    matrix = torch.normal(mean=0, std=1, size=(len(vocabulary), dimension))
    matrix[0] = torch.zeros(size=[dimension])
    if path is None:
        return Embedding.from_pretrained(matrix, padding_idx=0, freeze=freeze)
    data=pandas.read_csv(path, header=None, sep='\s+', engine='python')
    i=0
    for row in data:
        word = data.loc[i].loc[0]
        if word in vocabulary.stoi:
            embedding=[]
            for j in range (1, dimension):
                embedding.append(data.loc[i][j])
            matrix[vocabulary.stoi[word]] = torch.tensor(embedding)
        i+=1
    return Embedding.from_pretrained(matrix, padding_idx=0, freeze=freeze)

def pad_collate_fn(batch: tuple, pad_index: int=0):
    texts, labels = zip(*batch)
    lengths = torch.tensor([len(text) for text in texts])
    padded = pad_sequence(texts, batch_first=True, padding_value=pad_index)
    return padded, torch.tensor(labels), lengths


def train(model: torch.nn.Module, data: DataLoader, optimizer, path :Path, criterion):
    model.train()
    for batch_num, batch in enumerate(data):
        model.zero_grad()
        x, y, lens = batch
        logits = model(x).reshape(y.shape)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            pred = torch.sigmoid(logits)
        return (y, pred, loss.item())



def main():
    seed = 7052020
    np.random.seed(seed)
    torch.manual_seed(seed)

    trainData = pandas.read_csv(Path("./sst_train_raw.csv"), header=None)
    testData = pandas.read_csv(Path("./sst_test_raw.csv"), header=None)
    validationData = pandas.read_csv(Path("./sst_valid_raw.csv"), header=None)
    textFreq = frequencyDict(trainData[0])
    labelFreq = frequencyDict(trainData[1])
    textVocab = Vocab(textFreq, isTextArray=True, minFreq=1)
    labelVocab = Vocab(labelFreq, isTextArray=False, inFreq=1)
    train_dataset = NLPDataset(textVocab, labelVocab, trainData)
    train_loader = DataLoader(train_dataset, batch_size = 10, shuffle = True, collate_fn=pad_collate_fn)
    valid_dataset = NLPDataset(textVocab, labelVocab, validationData)
    valid_loader = DataLoader(valid_dataset, batch_size = 32, shuffle = False, collate_fn=pad_collate_fn)
    test_dataset = NLPDataset(textVocab, labelVocab, testData)
    test_loader =  DataLoader(test_dataset, batch_size = 32, shuffle = False, collate_fn=pad_collate_fn)

    embed = generateEmbeddingMatrix(path=Path("./glove.txt"), vocab=textVocab, dim=300, freeze= False)
    model = Baseline.Baseline(embed)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    allLoss = []
    confMatrix = np.zeros(shape=(2, 2))
    for epoch in range(30):
        y, pred, loss= train(model = model, data = train_loader, optimizer = optimizer, criterion =  criterion,path = Path("baselineLog.txt"))
        confMatrix+=confMatrix(y, pred)
        allLoss.append(loss)
        evaluate(...)


texts, labels, lengths = next(iter(train_loader))
print(f"Texts: {texts}")
print(f"Labels: {labels}")
print(f"Lengths: {lengths}")



