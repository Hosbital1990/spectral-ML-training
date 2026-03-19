import re
import subprocess
import sys
import time
from pathlib import Path

MODEL_SCRIPTS = [
    ("Random Forest", "step06_train_random_forest.py"),
    ("Extra Trees", "step06_train_extra_trees.py"),
    ("Linear SVM", "step06_train_linear_svm.py"),
    ("Logistic Regression", "step06_train_logistic_regression.py"),
    ("KNN", "step06_train_knn.py"),
    ("1D CNN", "step06_train_cnn_1d.py"),
]

ACCURACY_PATTERN = re.compile(r"Accuracy:\s*([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)")
LOG_DIR = Path("model_run_logs")


def extract_accuracy(output_text):
    match = ACCURACY_PATTERN.search(output_text)
    if match is None:
        return None
    return float(match.group(1))


def summarize_error(output_text):
    lines = [line.strip() for line in output_text.splitlines() if line.strip()]
    if not lines:
        return "No output"

    for line in reversed(lines):
        if "error" in line.lower() or "exception" in line.lower() or "traceback" in line.lower():
            return line[:100]

    return lines[-1][:100]


def format_seconds(seconds):
    return f"{seconds:.1f}s"


def build_table(rows):
    headers = ["Rank", "Model", "Status", "Accuracy", "Runtime", "Notes"]
    widths = [len(header) for header in headers]

    rendered_rows = []
    for row in rows:
        rendered_row = [
            str(row["rank"]),
            row["model"],
            row["status"],
            row["accuracy_text"],
            row["runtime_text"],
            row["notes"],
        ]
        rendered_rows.append(rendered_row)
        for index, value in enumerate(rendered_row):
            widths[index] = max(widths[index], len(value))

    divider = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    header_line = "| " + " | ".join(
        header.ljust(widths[index]) for index, header in enumerate(headers)
    ) + " |"

    lines = [divider, header_line, divider]
    for row in rendered_rows:
        lines.append(
            "| " + " | ".join(
                value.ljust(widths[index]) for index, value in enumerate(row)
            ) + " |"
        )
    lines.append(divider)
    return "\n".join(lines)


def main():
    project_root = Path(__file__).resolve().parent
    LOG_DIR.mkdir(exist_ok=True)

    print("Running model comparison suite...")
    print(f"Python interpreter: {sys.executable}")
    print(f"Logs directory: {LOG_DIR.resolve()}")

    results = []

    for model_name, script_name in MODEL_SCRIPTS:
        script_path = project_root / script_name
        log_path = LOG_DIR / f"{Path(script_name).stem}.log"

        print(f"\n=== Running {model_name} ===")
        print(f"Script: {script_name}")

        start_time = time.perf_counter()
        completed = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=project_root,
            capture_output=True,
            text=True,
        )
        elapsed = time.perf_counter() - start_time

        combined_output = completed.stdout
        if completed.stderr:
            combined_output += "\n--- STDERR ---\n" + completed.stderr

        log_path.write_text(combined_output, encoding="utf-8")

        accuracy = extract_accuracy(combined_output)
        status = "ok" if completed.returncode == 0 else "failed"
        notes = ""

        if completed.returncode != 0:
            notes = summarize_error(combined_output)
        elif accuracy is None:
            status = "unknown"
            notes = "Accuracy line not found"
        else:
            notes = log_path.name

        print(f"Status: {status}")
        print(f"Runtime: {format_seconds(elapsed)}")
        if accuracy is not None:
            print(f"Accuracy: {accuracy:.6f}")
        if notes:
            print(f"Notes: {notes}")

        results.append(
            {
                "model": model_name,
                "script": script_name,
                "status": status,
                "accuracy": accuracy,
                "accuracy_text": f"{accuracy:.6f}" if accuracy is not None else "n/a",
                "runtime_seconds": elapsed,
                "runtime_text": format_seconds(elapsed),
                "notes": notes,
            }
        )

    sorted_results = sorted(
        results,
        key=lambda item: (
            item["status"] != "ok",
            -(item["accuracy"] if item["accuracy"] is not None else -1.0),
            item["runtime_seconds"],
        ),
    )

    for index, item in enumerate(sorted_results, start=1):
        item["rank"] = index

    print("\n=== Final Comparison Table ===")
    print(build_table(sorted_results))

    successful_results = [item for item in sorted_results if item["accuracy"] is not None]
    if successful_results:
        best_result = successful_results[0]
        print(
            f"\nBest model: {best_result['model']} "
            f"with accuracy {best_result['accuracy']:.6f}"
        )
    else:
        print("\nNo model completed successfully.")


if __name__ == "__main__":
    main()
