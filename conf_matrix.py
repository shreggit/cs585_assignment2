from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    true_labels = []
    for i in range(50):
        if i % 5 == 0:
            true_labels.append('Closed Fist')
        elif i % 5 == 1:
            true_labels.append('Index')
        elif i % 5 == 2:
            true_labels.append('Open Palms')
        elif i % 5 == 3:
            true_labels.append('Thumbs Up')   
        else:
            true_labels.append('Thumbs Down')
    
    pred_labels = []
    for i in range(50):
        if i % 5 == 0:
            pred_labels.append('Closed Fist')
        elif i % 5 == 1:
            pred_labels.append('Index')
        elif i % 5 == 2 and (i!=22):
            pred_labels.append('Open Palms')
        elif i % 5 == 3 and (i!=23 and i!=28 and i!=43):
            pred_labels.append('Thumbs Up')   
        elif i%5 == 4:
            pred_labels.append('Thumbs Down')
        elif i == 28 or i==43:
            pred_labels.append('Index')
        elif i == 23:
            pred_labels.append('Open Palms')
        elif i == 22:
            pred_labels.append('Thumbs Up')

    cm = confusion_matrix(true_labels, pred_labels)
    labels = set(true_labels)
    #creating heatmap using seaborn
    sns.set(font_scale=1.2)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()
    plt.savefig('conf_matrix.png')


if __name__ == "__main__":
    main()