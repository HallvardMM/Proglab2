"""File for encrypting using different encryptions"""

from abc import ABC, abstractmethod
from random import randint
import crypto_utils


class Person(ABC):
    """abstract Superclass for sender, receiver and hacker of coded messages"""

    def __init__(self, key, cypher):
        self.key = key
        self.cypher = cypher

    def set_key(self, key):
        """function to set key"""
        self.key = key

    def get_key(self):
        """function to get key"""
        return self.key

    def set_cypher(self, cypher):
        """function to set cipher"""
        self.cypher = cypher

    @abstractmethod
    def operate_cypher(self, text):
        """abstract-method to operate cipher"""

    @abstractmethod
    def tostring(self):
        """method to print a nice output"""


class Sender(Person):
    """Class for sending encrypted texts"""

    def __init__(self, key, cypher):
        self.decoded_text = ""
        self.coded_text = ""
        Person.__init__(self, key, cypher)

    def operate_cypher(self, text):
        self.decoded_text = text
        self.coded_text = self.cypher.encode(self.key, self.decoded_text)
        return self.coded_text

    def send_cypher(self, receiver, text):
        """Sends the encrypted text
        if RSA is used then receiver has to generate keys"""
        if isinstance(self.cypher, RSA):
            receiver.rsa_generate_keys()
            self.key = receiver.rsa_get_public_key()
        receiver.receive_cypher(self.operate_cypher(text))

    def get_coded_text(self, text):
        """returns the coded text"""
        return self.operate_cypher(text)

    def tostring(self):
        print("Sender: text before encryption: " +
              str(self.decoded_text) +
              "\tText after encryption: " +
              str(self.coded_text) + "\n")


class Receiver(Person):
    """Class for receiving encrypted text"""

    def __init__(self, key, cypher):
        self.coded_text = ""
        self.decoded_text = ""
        self.public_key = None
        Person.__init__(self, key, cypher)

    def operate_cypher(self, text):
        self.coded_text = text
        self.decoded_text = self.cypher.decode(self.key, self.coded_text)
        return self.decoded_text

    def receive_cypher(self, coded_text):
        """Method for receiving the cypher"""
        self.operate_cypher(coded_text)

    def rsa_generate_keys(self):
        """Used for generating private and public keys
        as tuples for when using RSA
        """
        _p, _q = 0, 0
        _e = 0
        phi = 0
        check_gcd = 2
        while _p == _q or check_gcd != 1:
            _p = crypto_utils.generate_random_prime(8)
            _q = crypto_utils.generate_random_prime(8)
            phi = (_p - 1) * (_q - 1)
            _e = randint(3, phi - 1)
            check_gcd = gcd_check(_e, phi)
        _n = _p * _q
        _d = crypto_utils.modular_inverse(_e, phi)
        self.public_key = (_n, _e)
        print("recivers public key: " + str(self.public_key))
        self.key = (_n, _d)

    def tostring(self):
        print("Recevier: Text before decryption: " +
              str(self.coded_text) +
              "\tText after decryption: " +
              str(self.decoded_text) + "\n")

    def rsa_get_public_key(self):
        """Used for distributing public key when using RSA"""
        return self.public_key


class Hacker(Person):
    """Class to hack encrypted texts"""

    def __init__(self, cypher):
        self.english_words = [line.rstrip('\n')
                              for line in open('english_words.txt')]
        # self.english_words = [line.rstrip('\n')
        #                      for line in open('english_temp.txt')]
        #############################################################################################################
        self.likely_key = None
        self.decoded_text = ""
        self.coded_text = ""
        self.cypher = cypher

    def operate_cypher(self, text):
        """Beaware that during decryption of unbreakable hacker will do better with more words"""
        self.coded_text = text
        self.decoded_text = "*_*"
        possible_keys = self.possible_keys()
        for possible_key in possible_keys:
            decoded_text = self.cypher.decode(possible_key, text)
            for word in decoded_text.split():
                if word in self.english_words:
                    self.likely_key[possible_keys.index(possible_key)] += 1
        best_key = possible_keys[self.likely_key.index(max(self.likely_key))]
        print("Hacker found key: " + str(best_key))
        decoded_text = self.cypher.decode(best_key, text)
        self.decoded_text = decoded_text

    def possible_keys(self):
        """method that returns possible keys of
        different cryptation classes"""
        if isinstance(self.cypher, Caesar):
            self.likely_key = [0] * 95
            return [x for x in range(0, 95)]

        elif isinstance(self.cypher, Multiplicative):
            self.likely_key = [0] * 95
            return [x for x in range(0, 95)]

        elif isinstance(self.cypher, Affine):
            self.likely_key = [0] * 9025
            return [(x, y) for x in range(0, 95) for y in range(0, 95)]

        elif isinstance(self.cypher, Unbreakable):
            self.likely_key = [0] * len(self.english_words)
            return self.english_words

    def tostring(self):
        print("Hacker: Text before decryption: " +
              str(self.coded_text) +
              "\tText after decryption: " +
              str(self.decoded_text) + "\t")


class Cypher(ABC):
    """Abstract class for encryption"""

    def __init__(self):
        """ defines alphabet from space to ~ """
        self.alphabet = [chr(x) for x in range(32, 127)]

    @abstractmethod
    def encode(self, key, text):
        """abstract-method"""

    @abstractmethod
    def decode(self, key, text):
        """abstract-method"""

    def verify(self, key, text):
        """verifies if code has been encoded and decoded"""
        coded_text = self.encode(key, text)
        decoded_text = self.decode(key, coded_text)
        return text == decoded_text


class Caesar(Cypher):
    """Class to encode using Caesar coding """

    def encode(self, key, text):
        """Code letter: letter+key mod (length of alphabet)"""
        coded_text = ""
        for char in text:
            alphabet_place = (self.alphabet.index(
                char) + key) % len(self.alphabet)
            coded_text += self.alphabet[alphabet_place]
        return coded_text

    def decode(self, key, text):
        decoded_text = ""
        for char in text:
            alphabet_place = (self.alphabet.index(
                char) - key) % len(self.alphabet)
            decoded_char = self.alphabet[alphabet_place]
            decoded_text += decoded_char
        return decoded_text


class Multiplicative(Cypher):
    """Class to code text using multiplicative Caesar coding.
        Have to take into account that we need the modular inverse of the key"""

    def encode(self, key, text):
        """Code letter: letter*key mod (length of alphabet)"""
        coded_text = ""
        for char in text:
            alphabet_place = (self.alphabet.index(
                char) * key) % len(self.alphabet)
            coded_char = self.alphabet[alphabet_place]
            coded_text += coded_char
        return coded_text

    def decode(self, key, text):
        inverted_key = self.invert_key(key)
        return self.encode(inverted_key, text)

    def invert_key(self, key):
        """Method for creating the modular inverse of the key."""
        return crypto_utils.modular_inverse(key, len(self.alphabet))


class Affine(Cypher):
    """Coding using both caesar and multiplicative so we don't get A always coded as the same
    encoding: first Multiplicative, secondly Caesar
    decoding: first Caesar, secondly Multiplicative"""

    def encode(self, key, text):
        """The keys must be a tuple (k_1, k_2) of two numbers as specified in the task  """
        multi = Multiplicative()
        caesar = Caesar()
        return caesar.encode(key[1], multi.encode(key[0], text))

    def decode(self, key, text):
        """The keys must be a tuple (k_1, k_2) of two numbers as specified in the task"""
        multi = Multiplicative()
        caesar = Caesar()
        return multi.decode(key[0], caesar.decode(key[1], text))


class Unbreakable(Cypher):
    """Coding using a word as a key not a number.
       Using a word will help hide the natural flow of a language"""

    def encode(self, key, text):
        coded_text = ""
        temp_value = 0
        for char in text:
            alphabet_place = (self.alphabet.index(key[temp_value % len(key)]) +
                              self.alphabet.index(char)) % len(self.alphabet)
            coded_text += self.alphabet[alphabet_place]
            temp_value += 1
        return coded_text

    def decode(self, key, text):
        inverted_key = self.invert_key(key)
        return self.encode(inverted_key, text)

    def invert_key(self, key):
        """Method for finding the inverted key"""
        inverted_key = ""
        for char in key:
            inverted_key += self.alphabet[(len(self.alphabet) -
                                           self.alphabet.index(char)) %
                                          len(self.alphabet)]
        return inverted_key


class RSA(Cypher):
    """Coding using public and private keys. """

    def encode(self, key, text):
        """Takes inn key as a tuple (n, public key)"""
        _n = key[0]
        public_key = key[1]
        two_bits = crypto_utils.blocks_from_text(text, 2)
        coded_text = [pow(bit, public_key, _n) for bit in two_bits]
        return coded_text

    def decode(self, key, text):
        """Takes inn key as a tuple (n, private key)"""
        _n = key[0]
        private_key = key[1]
        two_bits = [pow(bit, private_key, _n) for bit in text]
        return crypto_utils.text_from_blocks(two_bits, 2)


def gcd_check(_e, phi):
    """Method from crypto_utils for checking if gcd is 1 or not"""
    previous_remainder, remainder = _e, phi
    current_x, previous_x, current_y, previous_y = 0, 1, 1, 0
    while remainder > 0:
        previous_remainder, (quotient, remainder) = remainder, divmod(
            previous_remainder, remainder)
        current_x, previous_x = previous_x - quotient * current_x, current_x
        current_y, previous_y = previous_y - quotient * current_y, current_y
    # The loop terminates with remainder == 0, x == b and y == -a.
    # This is not what we want, and is because we have
    # walked it through one time "too many". Therefore, return the values
    # of the previous round:
    return previous_remainder


def main():
    """main to run the program"""
    while True:
        choose_encryption = input(
            "Choose the preferred encryption [C/M/A/U/R]: ")
        if choose_encryption == "C":
            secret_text = "Cesar liked conventionalism"
            caesar = Caesar()

            sender = Sender(7, caesar)
            receiver = Receiver(sender.get_key(), caesar)
            hacker = Hacker(caesar)

            print("Testing Caesar")
            sender.send_cypher(receiver, secret_text)
            hacker.operate_cypher(sender.get_coded_text(secret_text))
            sender.tostring()
            receiver.tostring()
            hacker.tostring()

        elif choose_encryption == "M":
            secret_text = "more money more problems"
            multi = Multiplicative()

            sender = Sender(9, multi)
            receiver = Receiver(sender.get_key(), multi)
            hacker = Hacker(multi)

            print("Testing Multiplicative")
            sender.send_cypher(receiver, secret_text)
            hacker.operate_cypher(sender.get_coded_text(secret_text))
            sender.tostring()
            receiver.tostring()
            hacker.tostring()

        elif choose_encryption == "A":
            secret_text = "atomic power to the people"
            affine = Affine()

            sender = Sender((7, 3), affine)
            receiver = Receiver(sender.get_key(), affine)
            hacker = Hacker(affine)

            print("Testing Affine")
            sender.send_cypher(receiver, secret_text)
            hacker.operate_cypher(sender.get_coded_text(secret_text))
            sender.tostring()
            receiver.tostring()
            hacker.tostring()

        elif choose_encryption == "U":
            secret_text = "all alien avatars"
            unbreakable = Unbreakable()

            sender = Sender("another", unbreakable)
            receiver = Receiver(sender.get_key(), unbreakable)
            hacker = Hacker(unbreakable)

            print("Testing Unbreakable")
            print("OBS! Test will only check for letters starting with a!")
            sender.send_cypher(receiver, secret_text)
            hacker.operate_cypher(sender.get_coded_text(secret_text))
            sender.tostring()
            receiver.tostring()
            hacker.tostring()

        elif choose_encryption == "R":
            secret_text = "We are the world"
            rsa = RSA()

            sender = Sender("RSA", rsa)
            receiver = Receiver(sender.get_key(), rsa)

            print("Testing RSA")
            sender.send_cypher(receiver, secret_text)
            sender.tostring()
            receiver.tostring()
        else:
            print("Invalid input")


main()