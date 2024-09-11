# import KBManager
# import mainOntology
import DatasetPreProcessing
import DataAnalyzer
import SupervisedTrainingManager


# DatasetPreProcessing.call()
# DataAnalyzer.call()
TrainingManager.call()
'''
while True:
    print("\nCosa vuoi fare?\n1) Interagisci con la KB\n2) Interagisci con l'ontologia\n3) Esci\n")
    choice = input("Inserisci la tua scelta: ")

    if choice == '1':
        print("Caricando la KB... Attendi...\n")
        KBManager.main()
    elif choice == '2':
        mainOntology.main_ontology()
    elif choice == '3':
        print("Esco...")
        break
    else:
        print("Scelta non valida. Inserisci 1, 2 o 3.")
'''