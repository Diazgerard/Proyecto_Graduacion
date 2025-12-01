// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract MainContract {
    address public owner;
    string private data;

    event DataUpdated(address indexed by, string newData, uint256 timestamp);

    modifier onlyOwner() {
        require(msg.sender == owner, "No eres el propietario");
        _;
    }

    constructor() {
        owner = msg.sender;
    }

    function setData(string memory _data) public {
        data = _data;
        emit DataUpdated(msg.sender, _data, block.timestamp);
    }

    function getData() public view returns (string memory) {
        return data;
    }
}
