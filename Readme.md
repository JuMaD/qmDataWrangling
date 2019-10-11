# qMem Datawrangler
This script processes output of qMem measurement software into an Origin &reg; compatible *.csv files and matplotlib graphs to quickly visualy analyze data.
It also performs the following calculations:
- Absolute values of current / current density (for semi-log plots)
- Memory window (the ratio between the absolute values of odd and even sweeps)
- Differential resistance
- Fowler-Nordheim Plots
- Normalized differential resistance
- Statistics on all measured and calculated data (min, max, avg)    

## Calculations
TABLE OF PARAMETERS THAT CAN BE SET IN CONFIG

|  section  | parameter   | type  | explanation  |   
|---        |---          |---    |           ---|
| Directory |   home_directory |  Bool |  The directory where all folders with data can be found.\\Will automatically change to last one used and leaving it empty will set it to root    |
 |  Parameters |             |        |           | 
 |  Parameters | filter = Current [A]   |   |        |           |    
|  Parameters | resistance_slice = 0.1 | |        |           | 
|  Parameters | resistance_range = 0.4 |  |        |           | 
|  Parameters | mem_method = divide    |  |        |           | 
|  Parameters | start_index = 2        |  |        |           | 
  |  Parameters |             |        | |        |           |            |    
|  Calculate |             |        |           |   
    
|  Plot |             |        |           |   

[Directory]
home_directory = H:/_all/Software/DataWrangling_new

[Parameters]
filter = Current [A]
resistance_slice = 0.1
resistance_range = 0.4
mem_method = divide
start_index = 2

[Calculate]
absolute = True
stats = True
fowler_nordheim = True
memory_window = True
differential_resistance = True

[Plot]
currents = True
stats = True
fowler_nordheim = True
fn_stats = True
memory_window = True
resistance = True
resistance_slice = False

