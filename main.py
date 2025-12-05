from Utilities import video_capture
from model_handler import SignLanguageModel


def main():
    model = SignLanguageModel(model_path="./model/checkpoint-3806")
    def on_space_pressed(frame):
        print("\nAnalyzing sign...")
        try:
            result = model.analyze_sign(frame)
            print(f"{result}\n")
        except Exception as e:
            print(f"Error analyzing sign: {e}\n")

    video_capture(on_space_callback=on_space_pressed)


if __name__ == "__main__":
    main()