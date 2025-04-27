
import matplotlib.pyplot as plt

def analyze_lengths_and_thresholds(prompt_file, story_file, thresholds=[128, 256, 512, 1024]):
    prompt_lengths = []
    story_lengths = []
    
    with open(prompt_file, 'r', encoding='utf-8') as pf, open(story_file, 'r', encoding='utf-8') as sf:
        for prompt, story in zip(pf, sf):
            prompt_tokens = prompt.strip().lower().split()
            story_tokens = story.strip().lower().split()
            prompt_lengths.append(len(prompt_tokens))
            story_lengths.append(len(story_tokens))
    
    print(f"Prompt lengths: avg={sum(prompt_lengths)/len(prompt_lengths):.2f}, max={max(prompt_lengths)}, min={min(prompt_lengths)}")
    print(f"Story lengths: avg={sum(story_lengths)/len(story_lengths):.2f}, max={max(story_lengths)}, min={min(story_lengths)}")

    print("\nThreshold counts (number of samples exceeding thresholds):")
    for t in thresholds:
        prompts_over = sum([1 for l in prompt_lengths if l > t])
        stories_over = sum([1 for l in story_lengths if l > t])
        print(f"  >{t} tokens: {prompts_over} prompts, {stories_over} stories")
    
    return prompt_lengths, story_lengths

def plot_length_distribution(lengths, title):
    plt.figure(figsize=(10,6))
    plt.hist(lengths, bins=100, color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel('Number of Tokens')
    plt.ylabel('Number of Samples')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # CHANGE THESE if your paths are different
    prompt_file = "../data/original/train.wp_source"
    story_file = "../data/original/train.wp_target"

    # Analyze the lengths
    prompt_lengths, story_lengths = analyze_lengths_and_thresholds(prompt_file, story_file)

    # Plot the distributions
    plot_length_distribution(prompt_lengths, "Prompt Length Distribution")
    plot_length_distribution(story_lengths, "Story Length Distribution")