# Vivado Synthesis Script for FracNet_T
# Generated to fix synthesis flow issues

# 1. Set the part (Must match the HLS project setting)
set_part xczu3eg-sbva484-1-e

# 2. Set the design top module
set_property top FracNet_T [current_fileset]


# 3. Source IP generation scripts (Required for Floating Point IPs used by HLS)
set ip_tcl_files [glob -nocomplain "bnn.prj/solution1/syn/verilog/*_ip.tcl"]
foreach file $ip_tcl_files {
    puts "Sourcing IP script: $file"
    source $file
}

# 4. Read all generated Verilog files
# Using glob to include all files in the directory
# Ensure paths use forward slashes
set verilog_files [glob -nocomplain "bnn.prj/solution1/syn/verilog/*.v"]

if {[llength $verilog_files] == 0} {
    error "No Verilog files found in bnn.prj/solution1/syn/verilog/. Please run HLS C Synthesis first."
}

foreach file $verilog_files {
    read_verilog $file
}

# 4. Read XDC constraints (if any - none specified in original request, but good practice to allow placeholder)
# read_xdc constraints.xdc

# 5. Run Synthesis
# -mode out_of_context is often used for IP, but for full design synthesis usage default.
# The user asked for "RTL synthesis", typically implies `synth_design`.
synth_design -top FracNet_T -part xczu3eg-sbva484-1-e -mode out_of_context

# 6. Report utilization and timing
# Create a timestamped directory for reports to avoid overwriting
set timestamp [clock format [clock seconds] -format "%Y%m%d_%H%M%S"]
set report_dir "reports/run_$timestamp"
file mkdir $report_dir
puts "Saving reports to $report_dir"

report_utilization -file $report_dir/synthesis_utilization.rpt
report_timing_summary -file $report_dir/synthesis_timing.rpt

puts "Synthesis completed successfully."
