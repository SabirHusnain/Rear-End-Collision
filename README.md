# Rear-end Collision Impact Analysis using Experimental Sensor Data

## How to use
```pwsh
# Clone remote repository
git clone https://github.com/sabirhusnain577/Rear-End-Collision

# Open terminal in project directory
cd Rear-End-Collision

# Create virtual environment
pip3 install virtualenv
python3 -m venv env

# Activate environment
.\env\Scripts\activate

# Install dependencies
pip3 install -r requirements.txt

# Run python file
python3 '.\Rear End Collision Analysis.py'
```

## Instructions to download data from NHTSA database

- Visit web site [NHTSA Database](https://www.nhtsa.gov/research-data/research-testing-databases#/vehicle)
- Use filter:
  1. Impact Angle = 180 - 180
  2. Occupant Type = H3
- Download '''NHTSA EV5 ASCII X-Y''' format for each experiment performed.
- Extract each file in `./NHTSA/` directory.
- Rename each file like `{code}_{year}_{make}_{model}`, for example: `v05552_2005_TOYOTA_MATRIX`.
