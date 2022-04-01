## Invention Assignment

This project is an Invention Assignment that basically serves as a video Shazam. The code provides a script that can find the best match for a self-filmed video by comparing it to an existent database.

## Installation

Clone the repository into a folder on your personal machine, and make sure you have all of the necessary imports already installed.

## Usage

Open a new Terminal window and navigate to the folder where you cloned the repository. Run the following script to print the help tool:

```
python main.py -h
```

Afterwards, you can query the database with a script such as the following:

```
python main.py --file_path your_file_path --start your_start_time --end your_end_time --training_set your_training_set --feature chosen_feature
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update training and test videos as appropriate.