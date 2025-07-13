import re
from pathlib import Path

def parse_evaluation_file(file_path):
    completion_scores = []
    grasp_success_scores = []
    push_actions_per_completion = []
    episode_times = []

    with open(file_path, 'r') as f:
        episode_data = []

        for line in f:
            if line.startswith('End of Episode:'):
                match = re.search(
                    r'End of Episode: (\d+), .*?No of actions taken: (\d+), Time taken: ([\d.]+) seconds, Graspable: (\w+)',
                    line
                )
                if match:
                    episode_num = int(match.group(1))
                    num_actions = int(match.group(2))
                    time_taken = float(match.group(3))
                    graspable = match.group(4).lower() == 'true'

                    # Completion (C): episode ends in ≤5 actions
                    completion = 1 if num_actions < 6 else 0
                    completion_scores.append(completion)

                    # Grasp Success (GS): based on graspable flag
                    grasp_success = 1 if graspable else 0
                    grasp_success_scores.append(grasp_success)

                    # Motion Number (MN): for completed episodes only
                    if completion == 1:
                        push_actions_per_completion.append(num_actions)

                    episode_times.append(time_taken)
                    episode_data = []
            else:
                episode_data.append(line.strip())

    # Metrics calculation
    completion_rate = (sum(completion_scores) / len(completion_scores)) * 100 if completion_scores else 0
    grasp_success_rate = (sum(grasp_success_scores) / len(grasp_success_scores)) * 100 if grasp_success_scores else 0
    motion_number = sum(push_actions_per_completion) / len(push_actions_per_completion) if push_actions_per_completion else 0
    avg_time = sum(episode_times) / len(episode_times) if episode_times else 0

    return {
        'Completion (C)': f"{completion_rate:.1f}%",
        'Grasp Success (GS)': f"{grasp_success_rate:.1f}%",
        'Motion Number (MN)': f"{motion_number:.1f}",
        'Avg Time': f"{avg_time:.1f}s",
        'Episodes': len(completion_scores)
    }

def print_results_table(results):
    headers = ["Objects", "Completion (C)", "Grasp Success (GS)", "Motion Number (MN)", "Avg Time", "Episodes"]
    print("\nEvaluation Metrics Comparison:")
    print("-" * 90)
    print(f"{headers[0]:<15} {headers[1]:<15} {headers[2]:<15} {headers[3]:<15} {headers[4]:<15} {headers[5]}")
    print("-" * 90)

    for obj_count, metrics in results.items():
        print(f"{obj_count:<15} {metrics['Completion (C)']:<15} {metrics['Grasp Success (GS)']:<15} {metrics['Motion Number (MN)']:<15} {metrics['Avg Time']:<15} {metrics['Episodes']}")

    print("-" * 90)

def main():
    files = [
        'model19_eval_10objs.txt',
        'model19_eval_15objs.txt',
        'model19_eval_25objs.txt'
    ]

    results = {}

    for file in files:
        file_path = Path(file)
        if not file_path.exists():
            print(f"Warning: File {file} not found. Skipping...")
            continue

        try:
            # Extract number of objects using filename format: model19_eval_10objs.txt
            stem_parts = file_path.stem.split('_')
            obj_part = stem_parts[-1]  # e.g., "10objs"
            num_objs = int(obj_part.replace("objs", ""))
            metrics = parse_evaluation_file(file_path)
            results[f"{num_objs} objects"] = metrics
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")

    if not results:
        print("No valid evaluation files found.")
        return

    print_results_table(results)

    # Optional: write to file
    with open("evaluation_results.txt", "w") as f:
        f.write("Evaluation Metrics Comparison:\n")
        f.write("-" * 90 + "\n")
        f.write(f"{'Objects':<15} {'Completion (C)':<15} {'Grasp Success (GS)':<15} {'Motion Number (MN)':<15} {'Avg Time':<15} {'Episodes'}\n")
        f.write("-" * 90 + "\n")
        for obj_count, metrics in results.items():
            f.write(f"{obj_count:<15} {metrics['Completion (C)']:<15} {metrics['Grasp Success (GS)']:<15} {metrics['Motion Number (MN)']:<15} {metrics['Avg Time']:<15} {metrics['Episodes']}\n")
        f.write("-" * 90 + "\n")

if __name__ == "__main__":
    main()

# import re

# def parse_evaluation_file(file_path):
#     completion_scores = []
#     grasp_success_scores = []
#     push_actions_per_completion = []
#     episode_times = []

#     with open(file_path, 'r') as f:
#         for line in f:
#             line = line.strip()
#             if line.startswith('End of Episode:'):
#                 match = re.search(
#                     r'End of Episode: (\d+), .*?No of actions taken: (\d+), Time taken: ([\d.]+) seconds, Graspable: (\w+)',
#                     line
#                 )
#                 if match:
#                     episode_num = int(match.group(1))
#                     num_actions = int(match.group(2))
#                     time_taken = float(match.group(3))
#                     graspable = match.group(4).lower() == 'true'

#                     # Completion (C): if actions ≤ 5
#                     completion = 1 if num_actions < 5 else 0
#                     completion_scores.append(completion)

#                     # Grasp Success (GS): based on 'Graspable' value
#                     grasp_success = 1 if graspable else 0
#                     grasp_success_scores.append(grasp_success)

#                     # Motion Number (MN): number of actions in successful episodes
#                     if completion == 1:
#                         push_actions_per_completion.append(num_actions)

#                     episode_times.append(time_taken)

#     # Calculate metrics
#     completion_rate = (sum(completion_scores) / len(completion_scores)) * 100 if completion_scores else 0
#     grasp_success_rate = (sum(grasp_success_scores) / len(grasp_success_scores)) * 100 if grasp_success_scores else 0
#     motion_number = sum(push_actions_per_completion) / len(push_actions_per_completion) if push_actions_per_completion else 0
#     avg_time = sum(episode_times) / len(episode_times) if episode_times else 0

#     return {
#         'Completion (C)': f"{completion_rate:.1f}%",
#         'Grasp Success (GS)': f"{grasp_success_rate:.1f}%",
#         'Motion Number (MN)': f"{motion_number:.1f}",
#         'Avg Time': f"{avg_time:.1f}s",
#         'Episodes': len(completion_scores)
#     }

# def print_results_table(results):
#     headers = ["Objects", "Completion (C)", "Grasp Success (GS)", "Motion Number (MN)", "Avg Time", "Episodes"]
#     print("\nEvaluation Metrics Comparison:")
#     print("-" * 90)
#     print(f"{headers[0]:<15} {headers[1]:<15} {headers[2]:<15} {headers[3]:<15} {headers[4]:<15} {headers[5]}")
#     print("-" * 90)

#     for obj_count, metrics in results.items():
#         print(f"{obj_count:<15} {metrics['Completion (C)']:<15} {metrics['Grasp Success (GS)']:<15} {metrics['Motion Number (MN)']:<15} {metrics['Avg Time']:<15} {metrics['Episodes']}")

#     print("-" * 90)

# def main():
#     files = [
#         'model19_eval_10objs.txt',
#         'model19_eval_15objs.txt',
#         'model19_eval_25objs.txt'
#     ]

#     results = {}

#     for file in files:
#         try:
#             num_objs = int(re.search(r'model19_eval_(\d+)objs\.txt', file).group(1))
#             metrics = parse_evaluation_file(file)
#             results[f"{num_objs} objects"] = metrics
#         except FileNotFoundError:
#             print(f"Warning: File {file} not found. Skipping...")
#         except Exception as e:
#             print(f"Error processing {file}: {str(e)}")

#     if not results:
#         print("No valid evaluation files found.")
#         return

#     print_results_table(results)

#     # Save to file
#     with open("evaluation_results.txt", "w") as f:
#         f.write("Evaluation Metrics Comparison:\n")
#         f.write("-" * 90 + "\n")
#         f.write(f"{'Objects':<15} {'Completion (C)':<15} {'Grasp Success (GS)':<15} {'Motion Number (MN)':<15} {'Avg Time':<15} {'Episodes'}\n")
#         f.write("-" * 90 + "\n")
#         for obj_count, metrics in results.items():
#             f.write(f"{obj_count:<15} {metrics['Completion (C)']:<15} {metrics['Grasp Success (GS)']:<15} {metrics['Motion Number (MN)']:<15} {metrics['Avg Time']:<15} {metrics['Episodes']}\n")
#         f.write("-" * 90 + "\n")

# if __name__ == "__main__":
#     main()
