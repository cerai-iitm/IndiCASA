import pandas as pd

def main():
  """
  Download the IndiBias dataset, clean and transform its data, and save the processed data to a CSV file.
  """
  # URL to the sample dataset hosted on GitHub
  dataset_url = "https://raw.githubusercontent.com/sahoonihar/IndiBias/main/IndiBias_v1_sample.csv"

  # Load the dataset into a DataFrame
  data = pd.read_csv(dataset_url)

  # Remove unnecessary columns
  data = data.drop(columns=['Unnamed: 0', 'index'])

  # Create a boolean mask for rows with anti-stereotype examples
  anti_stereo_mask = data['stereo_antistereo'] == 'antistereo'

  # Swap the sentence columns for rows where the label is 'antistereo'
  data.loc[anti_stereo_mask, ['modified_eng_sent_more', 'modified_eng_sent_less']] = \
    data.loc[anti_stereo_mask, ['modified_eng_sent_less', 'modified_eng_sent_more']].values

  # Drop the original stereotype label column as it's no longer needed
  data = data.drop(columns=['stereo_antistereo'])

  # Export the cleaned and processed dataset to a CSV file without the index column
  output_file = 'IndiBias_Aligned_Stereotype.csv'
  data.to_csv(output_file, index=False)
  print(f"Processed data saved to {output_file}")

if __name__ == "__main__":
  main()