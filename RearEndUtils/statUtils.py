import pandas as pd
import matplotlib.pyplot as plt

def plotPak(path):
    df = pd.read_csv(path)

    plt.bar(df['Year'], df['Injured'], color='cyan',
            linestyle='-.', label='Injuries', width=0.05)
    plt.bar(df['Year'], df['Killed'], color='blue',
            linestyle='-', label='Deaths', width=0.05)
    plt.plot(df['Year'], df['Accidents']
             , color='red', linestyle='-', label='Accidents')
    plt.plot(df['Year'], df['Fatal'],
             color='green', linestyle='--', label='Fatalities')
    plt.plot(df['Year'], df['Non_Fatal'],
             color='black', linestyle='-.', label='No Fatalities')
    plt.plot(df['Year'], df['Vehiles'],
             color='orange', linestyle='-', label='Vechicles')
    plt.grid()
    plt.title('Trend for Pakistan statistics')
    plt.xlabel('Year')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Statistics/Pak_Plot.png')
    
def plotPoland(path):
    df = pd.read_csv(path)

    plt.bar(df['Year'], df['Deaths'], color='blue',
            linestyle='-', label='Deaths', width=0.25)
    plt.plot(df['Year'], df['Population'], color='red', linestyle='-', label='Population (0.001M)')
    plt.plot(df['Year'], df['Accidents'],
             color='green', linestyle='--', label='Accidents')
    plt.plot(df['Year'], df['Vehicles']/1000,
             color='black', linestyle='-.', label='Vehicles (0.001M)')
    plt.grid()
    plt.title('Trend for Poland statistics')
    plt.xlabel('Year')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Statistics/Poland_Plot.png')

def plotUSA(path):
    df = pd.read_csv(path)

    plt.bar(df['Year'], df['Deaths'], color='blue',
            linestyle='-', label='Deaths', width=0.25)
    plt.plot(df['Year'], df['VMT_Vehicle_Miles_Travelled_billions']
             * 10, color='red', linestyle='-', label='VMT (100M)')
    plt.plot(df['Year'], df['Fatalities_per_100_million_VMT']*10000,
             color='green', linestyle='--', label='Fatalities/VMT (0.1M)')
    plt.plot(df['Year'], df['Population_millions']*100,
             color='black', linestyle='-.', label='Population (0.1M)')
    plt.plot(df['Year'], df['Fatalities_per_100000_population']*1000,
             color='orange', linestyle='-', label='Fatalities/population (0.1M)')
    plt.grid()
    plt.title('Trend for USA statistics')
    plt.xlabel('Year')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Statistics/USA_Plot.png')
