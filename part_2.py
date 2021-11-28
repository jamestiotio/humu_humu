# Import libraries
import os
import numpy as np

# Set OS-independent paths, relative to current directory
es_dev_in_path = os.path.join("data", "ES", "dev.in")
es_dev_out_path = os.path.join("data", "ES", "dev.out")
es_train_path = os.path.join("data", "ES", "train")
ru_dev_in_path = os.path.join("data", "RU", "dev.in")
ru_dev_out_path = os.path.join("data", "RU", "dev.out")
ru_train_path = os.path.join("data", "RU", "train")
es_dev_p2_out_path = os.path.join("data", "ES", "dev.p2.out")


N = 7
START, O, BPOS, IPOS, BNEU, INEU, BNEG, INEG, END = 0, 1, 2, 3, 4, 5, 6, 7, 8
labels = {'START': START,
          'O': O,
          'B-positive': BPOS,
          'I-positive': IPOS,
          'B-neutral': BNEU,
          'I-neutral': INEU,
          'B-negative': BNEG,
          'I-negative': INEG,
          'END': END}
labels_list = ['START', 'O', 'B-positive', 'I-positive', 'B-neutral', 'I-neutral', 'B-negative', 'I-negative', 'END']

# Read dev.in data
def read_dev_in_data(filepath):
    results = []
    with open(filepath, "r", encoding="utf-8") as file:
        lines = file.readlines()
        for line in lines:
            # if line.strip():
            token = line.strip()
            results.append(token)
            # else:
            #     continue
    return results

# Read dev.out data
def read_dev_out_data(filepath):
    results = []
    with open(filepath, "r", encoding="utf-8") as file:
        lines = file.readlines()
        for line in lines:
            if len(line.strip().rsplit(" ", 1)) == 2:
                token, label = line.strip().rsplit(" ", 1)
                results.append((token, labels[label]))
            else:
                results.append(('', END))
                results.append(('', START))
    return results

data = read_dev_out_data(es_train_path)
# print(data)

# get transition matrix using MLE
def MLE_transition(data):
    countIJ = np.zeros([N+2, N+2])
    countY = np.zeros([N+2, 1])
    prev = START
    for pair in data:
        curr = pair[1]
        countY[prev] += 1
        countIJ[prev, curr] += 1
        prev = curr
        # prev = START if curr == END else curr
    # print(countIJ)
    # print(countY)
    # print(sum(countY))
    transition = countIJ / countY
    transition[N+1, 0] = 0
    return transition

transition = MLE_transition(data)

np.set_printoptions(precision=3, suppress=True)
# print(transition)

def viterbi(transition, emission, data):
    pi = [np.zeros([N+2, 1])]
    pi[0][0] = 1
    parents = []
    j = 0
    predicted_results = []
    for x in data:    
        if x == '':  # final step
            res = pi[j] * transition[:,-1:]
            output = []
            output.insert(0, np.argmax(res, axis = 0))
            print(np.argmax(res, axis = 0)[0])
            # break

            # output, trace back for the sequence
            while j > 1:
                j -= 1
                output.insert(0, parents[j][0,])

            # output to file
            for i in output:
                predicted_results.append(labels_list[int(i)])
            predicted_results.append('')

            # reset
            pi = [np.zeros([N+2, 1])]
            pi[0][0] = 1
            # parents = [np.zeros([N+2])]
            parents = []
            j = 0
                
        else: # propagate with viterbi
            b = np.reshape(np.pad(emission[:,0], 1), (-1, 1))
            # print(b)
            # print(pi[j])
            # print(np.matmul(pi[j], np.transpose(b)))
            res = np.matmul(pi[j], np.transpose(b)) * transition
            # print(res)
            pi.append(np.reshape(res.max(axis=0), (-1, 1)))
            # print(np.argmax(res, axis = 0))
            parents.append(np.argmax(res, axis = 0))
            # break
            j += 1

    with open(es_dev_p2_out_path, "w+", encoding="utf-8") as file:
        for line in predicted_results:
            file.write(line + "\n")
    return predicted_results


dev_in = read_dev_in_data(es_dev_in_path)
emission = np.random.rand(N, 50000)
# print(dev_in)
res = viterbi(transition, emission, dev_in)

print(len(dev_in), len(res))

        
