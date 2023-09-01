import pandas as pd
import matplotlib.pyplot as plt
import os
from os import error, listdir, name, terminal_size
from os.path import isfile, join
from numpy import average, mean, std

# Read and process log files
def read_accl_log_files(log_dir):
    accl_dataframes = []
    column_names = [
        "collective", "number_of_nodes", "rank_id", "number_of_banks",
        "size", "rx_buffer_size", "segment_size", "max_pkt_size",
        "execution_time", "throughput", "host", "protoc", "stack"
    ]

    for filename in os.listdir(log_dir):
        if filename.endswith(".log"):
            filepath = os.path.join(log_dir, filename)
            df = pd.read_csv(filepath, header=None)
            accl_dataframes.append(df)
    
    accl_dataframes_con = pd.concat(accl_dataframes)
    accl_dataframes_con.columns = column_names

    return accl_dataframes_con

def read_mpi_log_files(log_dir):
    
    mpi_dataframes = []
    
    column_names = [
        "collective", "count", "number_of_nodes", "fpga_to_host", "host_to_host",
        "host_to_host_send", "host_to_fpga"
    ]

    # Iterate through files in the directory
    for filename in os.listdir(log_dir):
        file_path = os.path.join(log_dir, filename)
        
        # Skip directories and only process regular files
        if not os.path.isfile(file_path):
            continue
        
        # skip the header as we append header later
        df = pd.read_csv(file_path, skiprows=1, header=None, names=column_names, delimiter=';')
        
        # Convert "broadcast" to "bcast" in the "collective" column; align with accl log name
        df["collective"] = df["collective"].apply(lambda x: "bcast" if x == "broadcast" else x)
        df["collective"] = df["collective"].apply(lambda x: "sendrcv" if x == "sendrecv" else x)
        
        # Add the "size" column by multiplying "count" by 4
        df["size"] = df["count"] * 4
        
        # Multiply certain columns by 1000000 to convert unit to us
        columns_to_multiply = ["fpga_to_host", "host_to_host", "host_to_host_send", "host_to_fpga"]
        for col in columns_to_multiply:
            df[col] = df[col] * 1000000
        
        mpi_dataframes.append(df)
    
    mpi_dataframes_con = pd.concat(mpi_dataframes, ignore_index=True)

    # Group by specified columns and calculate the mean for specific columns
    grouped_df = mpi_dataframes_con.groupby(["collective", "count", "number_of_nodes"]).mean()
    
    return grouped_df
    
# Generate plots
def generate_plots(accl_dataframes, mpi_dataframes, output_dir):
    collective_values = accl_dataframes['collective'].unique()
    line_styles = ['-', '--', '-.', ':']  # Different line styles for protoc values
    markers = ['o', 's', '^', 'D']  # Different markers for host values
    
    for collective in collective_values:
        accl_df = accl_dataframes[(accl_dataframes['collective'] == collective)]
        mpi_df = mpi_dataframes.loc[(mpi_dataframes.index.get_level_values("collective") == collective)]

        # plot the throughput of send/recv
        if collective == 'sendrecv':
            plt.figure()
            for host_idx, (host, host_group) in enumerate(accl_df.groupby('host')):
                for stack_idx, (stack, stack_group) in enumerate(host_group.groupby('stack')):   
                    for protoc_idx, (protoc, protoc_group) in enumerate(stack_group.groupby('protoc')):
                        accl_avg_tput = protoc_group.groupby('size')['throughput'].mean()
                        plt.plot(accl_avg_tput.index, accl_avg_tput,
                                label=f'{host}-{protoc}-{stack}',
                                linestyle=line_styles[protoc_idx % len(line_styles)],
                                marker=markers[host_idx % len(markers)])
            plt.xlabel('Size[B]')
            plt.ylabel('Throughput[Gbps]')
            plt.xscale('log', base=2)  # Set x-axis to log2 scale
            plt.title(f'Throughput vs Size for {collective}')
            plt.legend()
            plt.savefig(os.path.join(output_dir, f'{collective}_throughput.png'))
        
        # plot time-size for each collective and each node
        if collective != 'sendrecv':
            number_of_nodes_values = accl_df['number_of_nodes'].unique()
            for nodes in number_of_nodes_values:
                accl_filter_df = accl_df[accl_df['number_of_nodes'] == nodes]
                mpi_filter_df = mpi_df.loc[(mpi_df.index.get_level_values("number_of_nodes") == nodes)]
                # print(accl_filter_df)
                # print(mpi_filter_df)  

                # plot separate data for host and device
                for host_idx, (host, host_group) in enumerate(accl_filter_df.groupby('host')):    
                    
                    plt.figure()
                    # plot MPI RDMA lines with host data
                    if host == "host":
                        mpi_avg_time = mpi_filter_df['host_to_host'].reset_index(drop=True)
                        mpi_size = mpi_filter_df['size'].reset_index(drop=True) 
                        plt.plot(mpi_size, mpi_avg_time,
                                label=f'mpi-host-rdma',
                                linestyle=line_styles[0],
                                marker=markers[0])
                    
                    # # plot MPI RDMA lines with device data
                    if host == "device":
                        mpi_avg_time = (mpi_filter_df['host_to_host'] + mpi_filter_df['fpga_to_host'] + mpi_filter_df['host_to_fpga']).reset_index(drop=True)
                        mpi_size = mpi_filter_df['size'].reset_index(drop=True)
                        plt.plot(mpi_size, mpi_avg_time,
                                label=f'mpi-device-rdma',
                                linestyle=line_styles[1],
                                marker=markers[0])

                    # plot ACCL line
                    for stack_idx, (stack, stack_group) in enumerate(host_group.groupby('stack')):   
                        for protoc_idx, (protoc, protoc_group) in enumerate(stack_group.groupby('protoc')):
                            accl_avg_time = protoc_group.groupby('size')['execution_time'].mean()
                            plt.plot(accl_avg_time.index, accl_avg_time,
                                    label=f'accl-{host}-{protoc}-{stack}',
                                    linestyle=line_styles[protoc_idx % len(line_styles)],
                                    marker=markers[host_idx % len(markers)])
                            
                    # Process title/legend/fontsize
                    plt.xlabel('Size[B]')
                    plt.ylabel('Execution Time[us]')
                    plt.xscale('log', base=2)  # Set x-axis to log2 scale
                    plt.yscale('log', base=2)  # Set y-axis to log2 scale
                    plt.title(f'Execution Time vs Size for {collective} ({nodes} nodes, {host} data)')
                    plt.legend()
                    plt.savefig(os.path.join(output_dir, f'{collective}_rank_{nodes}_{host}_execution_time.png'))


if __name__ == "__main__":
    log_dir = "./accl_results"  # Update this to the directory containing your log files
    output_dir = "./plots/"
    mpi_log_dir = "./mpi-benchmarking/results-eth"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    accl_dataframes = read_accl_log_files(log_dir)
    mpi_dataframes = read_mpi_log_files(mpi_log_dir)
    # print(mpi_dataframes)
    # print(accl_dataframes)
    generate_plots(accl_dataframes, mpi_dataframes, output_dir)
