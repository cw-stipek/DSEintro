### Name: Clinton Stipek
### Fun Fact: Clinton likes to travel alot and used to live in a van picking fruit in New Zealand. 


def main():
    print("Would you like to know about Clinton's undergraduate, graduate school or his professional career?")
    print("Options: undergraduate, graduate, professional, personal")
    clinton_phase = input("Enter your choice: ").strip().lower()

    if clinton_phase == "undergraduate":
        print("\nClinton attended the University of Washington in Seattle where he was a student-athlete in swimming. He absolutely loves Seattle and it will always be home for him. His undergraduate degree was in Oceanography with a focus in geophysics and he thoroughly enjoyed his time there. Some of his favorite memories are biking around the San Juan Islands and camping.")
    elif clinton_phase == "graduate":
        print("\nAfter working for a few years after his undergraduate Clinton moved to Miami to attend the University of Miami for his MSc. He studied how seagrass communities react to freshwater pulses using machine learning and spatial analysis. He had a blast in Miami and really enjoyed the professional development. One of his favorite memories was going scuba diving in the Florida Keys.")
    elif clinton_phase == "professional":
        print("\nIn Clinton's professional career he is a research scientist at Oak Ridge National Laboratory where he works at the intersection of big data, artificial intelligence, and the built environment. His work helps to better inform population modeling, extreme events, the electrical grid infrastructure, among others. Some of his favorite memories thus far have been working in a highly collaborative setting and solving multi-faceted problems with his teammates.")
    elif clinton_phase =="personal":
        print("\nClinton loves to swim still in his personal time and also enjoys hiking, traveling, and spending time with his wonderful partner who keeps him sane but he also drives her crazy sometimes.")
    else:
        print("\nPlease enter the following options: 'undergraduate', 'graduate', 'professional'.")

if __name__ == "__main__":
    main()
