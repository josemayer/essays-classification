import matplotlib.pyplot as plt
import sys
import json

def main():
    filename = sys.argv[1]
    filename_without_ext = filename.split('.')[0]
    competence = filename.split('_')[1].split('.')[0]
    with open('../logs/' + filename, 'r') as f:
        lines = f.readlines()

        trial = []
        losses = []
        i = 1
        for line in lines:
            line = line.replace("\'", "\"")
            line = line.replace("\n", "")
            if line == '---':
                break
            elif line == '':
                continue
            else:
                data = json.loads(line)
                best_loss = data['val_loss']
                losses.append(best_loss)
                trial.append(i)
                i += 1

        plt.plot(trial, losses)
        plt.title(f"Otimização de Hiperparâmetros ({competence})")
        plt.ylabel('Perda em Validação')
        plt.xlabel('Trial')
        plt.savefig('../logs/plots/' + filename_without_ext + '.pgf')
        plt.clf()

        print(min(losses))

if __name__ == '__main__':
    main()
