
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def evaluate_model(model, test_loader, device, model_name, class_names):

    model.eval()

    actual_labels = []
    predicted_labels = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            # Binary classification with one output
            if outputs.ndim == 1 or outputs.shape[-1] == 1:
                predictions = (
                    torch.sigmoid(outputs.reshape(-1)) >= 0.5
                ).long()

            # Binary classification with two outputs
            else:
                predictions = torch.argmax(outputs, dim=1)

            actual_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predictions.cpu().numpy())

    accuracy = accuracy_score(actual_labels, predicted_labels)

    precision = precision_score(
        actual_labels, predicted_labels,
        zero_division=0
    )

    recall = recall_score(
        actual_labels, predicted_labels,
        zero_division=0
    )

    f1 = f1_score(
        actual_labels, predicted_labels,
        zero_division=0
    )

    matrix = confusion_matrix(
        actual_labels,
        predicted_labels,
        labels=[0, 1]
    )

    report = classification_report(
        actual_labels,
        predicted_labels,
        labels=[0, 1],
        target_names=class_names,
        zero_division=0
    )

    print("\n" + "=" * 55)
    print(f"{model_name.upper()} MODEL EVALUATION")
    print("=" * 55)

    print(f"Accuracy  : {accuracy * 100:.2f}%")
    print(f"Precision : {precision * 100:.2f}%")
    print(f"Recall    : {recall * 100:.2f}%")
    print(f"F1 Score  : {f1 * 100:.2f}%")

    print("\nConfusion Matrix:")
    print(matrix)

    print("\nClassification Report:")
    print(report)

    print("=" * 55)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": matrix
    }
