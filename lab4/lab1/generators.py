import random
import datetime


class PassportGenerator:
    def __init__(self):
        self.used_keys = {}

    def generate(self):
        key = self._generate_unique_key()
        self.used_keys[key] = (random.randrange(10**6 - 1))
        if key in self.used_keys:
            self.used_keys[key] += 1
        return "{:04d}".format(key) + " {:06d}".format(self.used_keys[key])

    def _generate_unique_key(self):
        key = (random.randrange(10**2 - 1) + 1) * 100 + (random.randrange(24))
        return key


class SnilsGenerator:
    def __init__(self):
        self.nums = ''

    def generate(self):
  
        self.nums = [
            random.randint(0, 9) if x != 3 and x != 7 and x != 11
            else '-' if x == 3 or x == 7
            else ' '
            for x in range(0, 12)
        ]

        cont = self.nums[0] * 9 + self.nums[1] * 8 + self.nums[2] * 7 + self.nums[4] * 6 + self.nums[5] * 5 + self.nums[6] * 4 + self.nums[8] * 3 + self.nums[9] * 2 + self.nums[10] * 1

        if cont in (100, 101):
            cont = '00'
        elif cont > 101:
            cont = cont % 101
            if cont in (100, 101):
                cont = '00'
            elif cont < 10:
                cont = '0' + str(cont)
        elif cont < 10:
            cont = '0' + str(cont)

        self.nums.append(str(cont))
        return ''.join([str(x) for x in self.nums])


class NamesGenerator:
    def __init__(self, names, lastnames, patronymics):
        self.names = names
        self.lastnames = lastnames
        self.patronymics = patronymics

    def generate(self):
        return " ".join(
            [
                random.choice(self.lastnames),
                random.choice(self.names),
                random.choice(self.patronymics),
            ]
        )


class SamplesGenerator:
    def __init__(self, symptoms, k):
        self.symptoms = symptoms
        self.k = k

    def generate(self):
        return "|".join(random.sample(self.symptoms, k=self.k))


class DatetimeGenerator:
    def __init__(self):
        self.last = 0

    def generate(self):
        return (self._generate_date() + self._generate_time()).strftime(
            "%Y-%m-%dT%H:%M+03:00"
        )

    def _generate_date(self):
        if self.last == 0:
            rnd_days_ago = random.randrange(300)
            date = datetime.datetime.now() - datetime.timedelta(days=rnd_days_ago)
            if date.weekday() >= 5:
                date = date - datetime.timedelta(days=2)
            self.last = date
            return date
        else:
            date = self.last + datetime.timedelta(days=1)
            if date.weekday() >= 5:
                date = self.last + datetime.timedelta(days=2)
            self.last = 0
            return date

    def _generate_time(self):
        time = datetime.timedelta(minutes=(random.randrange(-180, 360)))
        return time


class CardGenerator:
    def __init__(self, banks, systems, keys):
        self.banks = banks
        self.systems = systems
        self.keys = keys
        self.used = {}

    def _generate_key(self, p_dict):
        rnd = random.randrange(0, 100)
        accum = 0
        for key in p_dict:
            accum += p_dict[key]
            if rnd <= accum:
                return key
        return ""

    def generate(self):
        bank = self._generate_key(self.banks)
        system = self._generate_key(self.systems)
        key = str(self.keys[bank + "_" + system])

        self.used[key] = (random.randrange(10**10 - 1))

        card = key + "{:010d}".format(self.used[key])
        card_split = []
        for i in range(4):
            card_split.append(card[i * 4 : (i + 1) * 4])
        return " ".join(card_split)
