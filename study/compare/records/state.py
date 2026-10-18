# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# @@ a type that holds state
class Account:
    def __init__(self, owner, balance):
        self.owner = owner
        self.balance = balance
        self.history = []

    # @@ operations that change it
    def deposit(self, amount):
        self.balance += amount
        self.history.append(amount)

    def withdraw(self, amount):
        if amount > self.balance:
            print(f"refused: {self.owner} has {self.balance:.2f}, asked for {amount:.2f}")
            return
        self.balance -= amount
        self.history.append(-amount)

# @@ use it
acct = Account("Ada", 100.0)
acct.deposit(25.5)
acct.withdraw(40.0)
acct.withdraw(500.0)
print(f"{acct.owner}: {acct.balance:.2f}")
print("history:", acct.history)

# @@ checks
assert acct.balance == 85.5
assert len(acct.history) == 2
