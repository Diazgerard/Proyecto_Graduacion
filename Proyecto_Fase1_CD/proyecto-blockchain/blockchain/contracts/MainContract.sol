// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract MainContract {
    // Variables principales
    address public owner;
    string public data;
    uint256 public transactionCount;
    
    // Estructura para transacciones
    struct Transaction {
        address from;
        address to;
        uint256 amount;
        string message;
        uint256 timestamp;
    }
    
    // Mappings
    mapping(address => uint256) public balances;
    mapping(uint256 => Transaction) public transactions;
    
    // Eventos
    event DataUpdated(string newData, address updatedBy);
    event TransactionStored(address from, address to, uint256 amount, string message);
    event BalanceUpdated(address user, uint256 newBalance);
    
    // Modificadores
    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can call this function");
        _;
    }
    
    // Constructor
    constructor() {
        owner = msg.sender;
        data = "Initial data";
        transactionCount = 0;
    }
    
    // Funciones principales
    function setData(string memory _data) public {
        data = _data;
        emit DataUpdated(_data, msg.sender);
    }

    function getData() public view returns (string memory) {
        return data;
    }
    
    function storeTransaction(
        address _to,
        uint256 _amount,
        string memory _message
    ) public {
        transactions[transactionCount] = Transaction(
            msg.sender,
            _to,
            _amount,
            _message,
            block.timestamp
        );
        
        emit TransactionStored(msg.sender, _to, _amount, _message);
        transactionCount++;
    }
    
    function getTransaction(uint256 _index) public view returns (
        address from,
        address to,
        uint256 amount,
        string memory message,
        uint256 timestamp
    ) {
        require(_index < transactionCount, "Transaction does not exist");
        Transaction memory transaction = transactions[_index];
        return (
            transaction.from,
            transaction.to,
            transaction.amount,
            transaction.message,
            transaction.timestamp
        );
    }
    
    function updateBalance(address _user, uint256 _balance) public onlyOwner {
        balances[_user] = _balance;
        emit BalanceUpdated(_user, _balance);
    }
    
    function getBalance(address _user) public view returns (uint256) {
        return balances[_user];
    }
    
    function getAllTransactions() public view returns (Transaction[] memory) {
        Transaction[] memory allTransactions = new Transaction[](transactionCount);
        for (uint256 i = 0; i < transactionCount; i++) {
            allTransactions[i] = transactions[i];
        }
        return allTransactions;
    }
    
    function executeAction(string memory _action) public returns (string memory) {
        // Función genérica para ejecutar acciones
        emit DataUpdated(_action, msg.sender);
        return string(abi.encodePacked("Action executed: ", _action));
    }
}
