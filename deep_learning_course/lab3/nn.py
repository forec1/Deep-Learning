import torch
import numpy as np
import data
import baseline_model
import rnn_model
import argparse
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter


def train(model, data, optimizer, criterion, device, args):
    model.train()
    for batch_num, batch in enumerate(data):
        model.zero_grad()
        optimizer.zero_grad()

        x, y, lenghts = batch
        x, y = x.to(device), y.to(device)
        y = y.reshape(-1,1).type(torch.float32)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()


def evaluate(model, data, criterion, device):
    model.eval()
    loss_avg, correct = 0.0, 0
    confusion_m = np.zeros((2, 2), dtype=np.int32)
    with torch.no_grad():
        for batch_num, batch in enumerate(data):
            x, y, lenghts = batch
            x, y = x.to(device), y.to(device)
            y = y.reshape(-1,1).type(torch.float32)
            logits = model(x)
            y_ = (torch.sigmoid(logits) >= 0.5).type(torch.uint8)

            loss_avg += criterion(logits, y).item()
            y = y.flatten().type(torch.uint8)
            y_ = y_.flatten()
            confusion_m += confusion_matrix(y.cpu(), y_.cpu(), labels=range(2))
            correct += (y == y_).sum().item()

    N = len(data) * data.batch_size
    num_batches = N // data.batch_size

    acc = correct / N * 100
    loss_avg = loss_avg / num_batches

    TP = confusion_m[0,0]
    FP = confusion_m[0,1]
    FN = confusion_m[1,0]
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * (precision * recall) / (precision + recall)

    return acc, loss_avg, confusion_m, f1


def print_metrics(metrics, mode): 
    print(mode+":")
    print("Accuracy = %.3f\nAverage loss = %.3f\nF1 metric = %.3f"
          %(metrics[0], metrics[1], metrics[3]))
    print("Confusion matrix:")
    print(metrics[2])
    print()


def main(args):
    if args.seed:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    train_dataset = data.NLPDataset.from_file('./data/sst_train_raw.csv', max_size=args.vocab_size, min_freq=args.min_freq)
    test_dataset = data.NLPDataset.from_file('./data/sst_test_raw.csv', create_vocab=False)
    val_dataset = data.NLPDataset.from_file('./data/sst_valid_raw.csv', create_vocab=False)
    test_dataset.text_vocab = train_dataset.text_vocab
    test_dataset.label_vocab = train_dataset.label_vocab
    val_dataset.text_vocab = train_dataset.text_vocab
    val_dataset.label_vocab = train_dataset.label_vocab

    file_path = None if args.no_pretrained_vec else './data/sst_glove_6b_300d.txt'
    embedded_matrix = data.generate_embedding_matrix(
        train_dataset.text_vocab, file_path=file_path)

    if args.model == 'base':
        model = baseline_model.BaseLineModel(embedded_matrix)
    else:
        if args.activation_fn == 'relu':
            act_fn = torch.nn.functional.relu
        elif args.activation_fn == 'tanh':
            act_fn = torch.tanh
        elif args.activation_fn == 'sigmoid':
            act_fn = torch.sigmoid

        model = rnn_model.MyRNN(embedded_matrix, h=args.hidden_size, dropout=args.dropout,
                                bidirectional=args.bidir, num_layers=args.num_layers, activation_fn=act_fn,
                                rnn=args.model)

    criterion = torch.nn.BCEWithLogitsLoss()
    if args.opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    elif args.opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

    train_data = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             collate_fn=data.pad_collate_fn)
    test_data = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=32,
                                            shuffle=False,
                                            collate_fn=data.pad_collate_fn)
    val_data = torch.utils.data.DataLoader(val_dataset,
                                           batch_size=32,
                                           shuffle=False,
                                           collate_fn=data.pad_collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available and args.cuda else "cpu")
    model.to(device)

    for epoch in range(args.epochs):
        train(model, train_data, optimizer, criterion, device, args)
        metrics = evaluate(model, val_data, criterion, device)
        print("Epoch %d:" % (epoch+1), end=' ')
        print_metrics(metrics, "Validation")

    metrics = evaluate(model, test_data, criterion, device)
    print_metrics(metrics, "Test")

    with SummaryWriter(log_dir=args.log_dir) as writer:
        writer.add_hparams({'seed': args.seed, 'epochs': args.epochs,
                            'cuda': args.cuda, 'batch_size': args.batch_size,
                            'model': args.model, 'dropout': args.dropout,
                            'bidirectional': args.bidir, 'hidden_size': args.hidden_size,
                            'num_layers': args.num_layers, 'activation_fn': args.activation_fn,
                            'vocab_size': args.vocab_size, 'min_freq': args.min_freq,
                            'opt': args.opt, 'clip': args.clip},
                           {'test/accuracy': metrics[0], 'test/avg_loss': metrics[1],
                            'test/f1': metrics[3]})


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate baseline model.")
    parser.add_argument("--seed", type=int, help="Seed.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs; default=5")
    parser.add_argument("-c", "--cuda", action="store_true", help="If specified CUDA tensors will be used.")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for training; default=10")
    parser.add_argument("--model", default='base', choices=['vanilla', 'lstm', 'gru', 'base'], help="Model type; default=base")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout probability; default=0.0")
    parser.add_argument("--bidir", action="store_true", help="If specified bidirectional rnn is used.")
    parser.add_argument("--hidden_size", default=150, type=int, help="Number of features in the hidden state h of rnn;default=150")
    parser.add_argument("-nl", "--num_layers", default=2, type=int, help="Number of recurrent layers.")
    parser.add_argument("-af", "--activation_fn", default='relu', choices=['relu', 'tanh', 'sigmoid'], help="Activation function for fully connected layers;default=relu")
    parser.add_argument("--vocab_size", type=int, default=-1, help="Vocabulary size;default=-1")
    parser.add_argument("--min_freq", type=int, default=0, help="Minimal frequency used when creating vocabulary;default=0")
    parser.add_argument("--opt", default='adam',choices=['adam', 'sgd'], help="Optimization algorithm;default=adam")
    parser.add_argument("--clip", type=float, default=0.25)
    parser.add_argument("--no_pretrained_vec", action="store_true")
    parser.add_argument("--log_dir", default="base")
    args = parser.parse_args()
    main(args)
