# NLU Course Project
This repository is dedicated to the implementation of the course project for the Natural Language Understanding (NLU) course at the University of Trento for the 2024/2025 academic year.

For more information about the tasks to accomplish you can refer to labs 4 and 5 of the [following repo](https://github.com/BrownFortress/NLU-2025-Labs).

## Setup
This project uses [`uv`](https://docs.astral.sh/uv/) as its package manager and [Hydra](https://hydra.cc/) to to organize and track experiment configurations.

To run the project, first clone the repository
```bash
git clone https://github.com/RobertoSavi/NLU-RS-Project.git
```

Then navigate to the project's folder, create and activate the uv virtual environment, and install the dependencies:
```bash
cd NLU-RS-Project
uv venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
uv sync
```

## Project Structure
The project is divided in two parts:
- **LM** (Language Modeling) focused on next-word prediction.
- **NLU** (Natural Language Understanding) focused on intent classification and slot filling.
A separate folder is dedicated for each section, both further divided into two parts as follows:
```text
NLU-RS-Project/
├── LM/
│   ├── part_A/
│   └── part_B/
├── NLU/
│   ├── part_A/
│   └── part_B/
```
A report describing the task, implementation and results is included for each part.
Inside each sub-folder a README will provide further instructions on how to run the code.

## Contact
Roberto Savi ([roberto.savi@studenti.unitn.it](mailto:roberto.savi@studenti.unitn.it))