from utils import get_data, process_data

def main():
    df = get_data()
    process_data(df)
    print("Finished processing data.")

if __name__ == "__main__":
    main()
