import random
import pandas
from generators import *


def compute(n, out_file, banks_p, systems_p):
    dictPath = "dictionary"

    names1 = [
        i.replace("\n", "")
        for i in (open(dictPath + "/names_1.txt", "r", encoding="utf-8").readlines())
    ]
    names2 = [
        i.replace("\n", "")
        for i in (open(dictPath + "/names_2.txt", "r", encoding="utf-8").readlines())
    ]

    lastnames1 = [
        i.replace("\n", "")
        for i in (
            open(dictPath + "/lastnames_1.txt", "r", encoding="utf-8").readlines()
        )
    ]
    lastnames2 = [
        i.replace("\n", "")
        for i in (
            open(dictPath + "/lastnames_2.txt", "r", encoding="utf-8").readlines()
        )
    ]

    patronymics1 = [
        i.replace("\n", "")
        for i in (
            open(dictPath + "/patronymics_1.txt", "r", encoding="utf-8").readlines()
        )
    ]
    patronymics2 = [
        i.replace("\n", "")
        for i in (
            open(dictPath + "/patronymics_2.txt", "r", encoding="utf-8").readlines()
        )
    ]

    doctors = [
        i.replace("\n", "")
        for i in (open(dictPath + "/doctors.txt", "r", encoding="utf-8").readlines())
    ]

    symptoms = [
        i.replace("\n", "")
        for i in (open(dictPath + "/symptoms.txt", "r", encoding="utf-8").readlines())
    ]

    symptoms_specific = [
        i.replace("\n", "")
        for i in (open(dictPath + "/symptoms_specific.txt", "r", encoding="utf-8").readlines())
    ]

    analysis = [
        i.replace("\n", "")
        for i in (open(dictPath + "/analysis.txt", "r", encoding="utf-8").readlines())
    ]

    analysis_specific = [
        i.replace("\n", "")
        for i in (open(dictPath + "/analysis_specific.txt", "r", encoding="utf-8").readlines())
    ]

    card_keys = {}
    for i in open(dictPath + "/card_keys.txt", "r", encoding="utf-8").readlines():
        row = i.replace("\n", "").split(" ")
        card_keys[row[0] + "_" + row[1]] = row[2]

    namesGen = [
        NamesGenerator(names1, lastnames1, patronymics1),
        NamesGenerator(names2, lastnames2, patronymics2),
    ]

    passportGen = PassportGenerator()
    snilsGen = SnilsGenerator()


    sympGen = SamplesGenerator(symptoms, 1)
    analysisGen = SamplesGenerator(analysis, 1)

    dateGen = DatetimeGenerator()

    cardGen = CardGenerator(banks_p, systems_p, card_keys)

    data = {
        "Name": [],
        "Passport": [],
        "Snils": [],
        "Symptoms": [],
        "Analysis": [],
        "Doctor": [],
        "DateStart": [],
        "DateEnd": [],
        "Price": [],
        "Card": [],
    }

    datacheck = {
        "Name": [],
        "Passport": [],
        "Snils": [],
        "Card" : [],
    }


    for i in range(n):
        data["Name"].append(namesGen[random.randrange(2)].generate()) 
        data["Passport"].append(passportGen.generate())
        snilscheck = snilsGen.generate()
        while snilscheck in data["Snils"]:
            snilscheck = snilsGen.generate()
        data["Snils"].append(snilscheck)

    for i in range(n):
        
        if data["Name"][i] in datacheck["Name"]:
            ind = datacheck["Name"].index(data["Name"][i])
            datacheck["Name"].append(data["Name"][ind])
            datacheck["Passport"].append(data["Passport"][ind])
            datacheck["Snils"].append(data["Snils"][ind])

        else:
            datacheck["Name"].append(data["Name"][i])
            datacheck["Passport"].append(data["Passport"][i])
            datacheck["Snils"].append(data["Snils"][i])

    data["Name"] = datacheck["Name"]
    data["Passport"] = datacheck["Passport"]
    data["Snils"] = datacheck["Snils"]

        

    for i in range(n):  
        num = random.randint(0, 49)  
        data["Symptoms"].append(sympGen.generate() + "|" + symptoms_specific[num])
        data["Analysis"].append(analysisGen.generate() + "|" + analysis_specific[num])
        data["Doctor"].append(doctors[num])
        data["DateStart"].append(dateGen.generate())
        data["DateEnd"].append(dateGen.generate())
        data["Price"].append(str(random.randint(10, 100) * 100))

    for i in range(n):
        data["Card"].append(cardGen.generate())

    for i in range(n):
        if (data["Name"][i] in datacheck["Name"]):
            ind1 = datacheck["Name"].index(data["Name"][i])
            datacheck["Card"].append(data["Card"][ind1])

        else:
            datacheck["Card"].append(data["Card"][i])

    data["Card"] = datacheck["Card"]


    df = pandas.DataFrame(data)
    df.to_csv(f"{out_file}.csv", index=False)
