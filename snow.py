import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

datafile = pd.read_csv("snow_instability_field_data.csv", sep=";").dropna(how="all")

# return true (unstable) if score is 1, 2, or 3 (very poor, poor, fair)
def five_to_binary(score):
    if not (1 <= score <= 5):
        raise ValueError("Score out of bounds")
    return score < 4

# return true (unstable) if score is 1 or 2 (very poor, poor)
def techel_to_binary(score):
    if not (1 <= score <= 4):
        raise ValueError("Score out of bounds")
    return score < 3

# return true (unstable) if score is 2 or 3 (number of criteria in critical range)
def three_to_binary(score):
    if not (1 <= score <= 3):
        raise ValueError("Score out of bounds")
    return score > 1

binarizers = {
    "5-class_Stability": five_to_binary,
    "4-class_Stability [Techel]": techel_to_binary,
    "3-class_Stability [sum S2008: 1+2+3]": three_to_binary
}

for scheme in binarizers:
    assert scheme in datafile.columns
assert "Avalanche_activity" in datafile.columns

y_true = datafile["Avalanche_activity"]
numAvals = (y_true == 1).sum()
numNotAvals = (y_true == 0).sum()
assert numAvals + numNotAvals == 589

#results = {}
labels = [1, 0]
for name in binarizers:
    y_pred = datafile[name].map(binarizers[name])

    #numBothAval = ((datafile[name] < 3) & (y_true == 1)).sum()
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    disp = ConfusionMatrixDisplay(cm, display_labels=["Avalanche", "No Avalanche"])
    ax = disp.plot(cmap="Blues", values_format="d")
    ax.ax_.set_title(name)
    plt.tight_layout()
    plt.show()

    print(f"\n{name} - confusion matrix (rows = actual, cols = predicted)")
    print(cm)
    print(classification_report(y_true, y_pred, labels=labels, target_names=["Avalanche","No Avalanche"]))

    #results[name] = {"cm": cm}
