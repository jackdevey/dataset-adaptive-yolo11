from ultralytics import YOLO

from utils.notifications.telegram import TelegramAPI


def add_training_callbacks(model: YOLO, telegram: TelegramAPI):
    """Callback function to send a message when the training starts."""

    def cb_train_start(trainer):
        telegram.message("🚀 Training started!")

    def cb_train_end(trainer):
        telegram.message("🎉 Training finished!")

    def cb_epoch_start(trainer):
        epoch = trainer.epoch
        telegram.message(f"🔄 Starting epoch {epoch}...")

    def cb_epoch_finished(trainer):
        epoch = trainer.epoch
        metrics = trainer.metrics
        results_summary = f"📊 Epoch {epoch} finished!\n"
        if metrics:
            # Format key metrics in a structured way
            key_metrics = {
                "box_loss": metrics.get("box_loss", "N/A"),
                "cls_loss": metrics.get("cls_loss", "N/A"),
                "dfl_loss": metrics.get("dfl_loss", "N/A"),
                "precision": metrics.get("precision", "N/A"),
                "recall": metrics.get("recall", "N/A"),
                "mAP50": metrics.get("mAP50", "N/A"),
                "mAP50-95": metrics.get("mAP50-95", "N/A"),
            }
            # Format each metric on a new line with proper escaping
            metrics_str = "\n".join(
                [
                    f"{k}: {v:.4f}" if isinstance(v, (int, float)) else f"{k}: {v}"
                    for k, v in key_metrics.items()
                ]
            )
            # Add the metrics to the summary
            results_summary += metrics_str
        telegram.message(results_summary)

    model.add_callback("on_train_start", cb_train_start)
    model.add_callback("on_train_end", cb_train_end)
    model.add_callback("on_train_epoch_start", cb_epoch_start)
    model.add_callback("on_fit_epoch_end", cb_epoch_finished)
