# -*- coding: utf-8 -*-
import hashlib
import json
from time import time
from  uuid import uuid4
from textwrap import dedent

class blockchain(object):
    def __init__(self):
        self.chain=[]
        self.current_transactions=[]
        #create the genesis block
        self.new_block(previous_hash=1,proof=100)
    def new_block(self,proof,previous_hash=None):
        block={
            'index':len(self.chain)+1,
            'timestamp':time(),
            'transactions':self.current_transactions,
            'proof':proof,
            'previous_hash':previous_hash or self.hash(self.chain[-1]),
        }
        #Reset the current list of transactions
        self.current_transactions=[]
        self.chain.append(block)
        return block
    def new_transaction(self,sender,recipient,amount):
        """
        #create a new transactions to go into the next mined block
        :param sender:
        :param recipient:
        :param amount:
        :return:
        """
        self.current_transactions.append({
            'sender':sender,
            'recipient':recipient,
            'amount':amount,
        })
        return self.last_block['index']+1
    @property
    def last_block(self):
        return self.chain[-1]
    @staticmethod
    def hash(block):
        """
        #create a sha-256 hash of a block
        :param block:
        :return:
        """
        block_string=json.dumps(block,sort_keys=True).encode()
        return hashlib.sha3_256(block_string).hexdigest()

    def proof_of_work(self,last_proof):
        proof=0
        while self.valid_proof(last_proof,proof)is False:
            proof+=1
        return proof
    def valid_proof(last_proof,proof):
        guess=f'{last_proof}{proof}'.encode()
        guess_hash=hashlib.sha256(guess).hexdigest()
        return guess_hash[:4]=="0000"
    #app=Flask(__name__)
