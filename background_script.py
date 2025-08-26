


def main():
    print("Would you like to know about Clinton's undergraduate, graduate school or his professional career?")
    print("Options: undergraduate, graduate, professional")
    clinton_phase = input("Enter your choice: ").strip().lower()

    if clinton_phase == "undergraduate":
        print("\nClinton attended the University of Washington in Seattle where he was a student-athlete in swimming. He absolutely loves Seattle and it will always be home for him. His undergraduate degree was in Oceanography with a focus in geophysics and he thoroughly enjoyed his time there.")
    elif clinton_phase == "graduate":
        print("\nAfter working for a few years after his undergraduate Clinton moved to Miami to attend the University of Miami for his MSc. He studied how seagrass communities react to freshwater pulses using machine learning and spatial analysis. He had a blast in Miami and really enjoyed the professional development")
    elif clinton_phase == "professional":
        print("\nIn Clinton's professional career he is a research scientist at Oak Ridge National Laboratory where he works at the intersection of big data, artificial intelligence, and the built environment. His work helps to better inform population modeling, extreme events, the electrical grid infrastructure, among others.")
    else:
        print("\nPlease enter the following options: 'undergraduate', 'graduate', 'professional'.")

if __name__ == "__main__":
    main()
