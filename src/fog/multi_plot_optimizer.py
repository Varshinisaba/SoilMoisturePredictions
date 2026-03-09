plots = {
"Plot1":35,
"Plot2":55,
"Plot3":25
}

def schedule_irrigation(plots):

    sorted_plots = sorted(plots.items(),key=lambda x:x[1])

    print("Irrigation priority:")

    for p in sorted_plots:
        print(p[0],"Moisture:",p[1])

schedule_irrigation(plots)