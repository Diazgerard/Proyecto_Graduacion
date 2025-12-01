const { ethers } = require("hardhat");
const fs = require("fs");

async function main() {
  const Contract = await ethers.getContractFactory("MainContract");
  const contract = await Contract.deploy();
  await contract.waitForDeployment();

  const address = await contract.getAddress();
  console.log("Contrato desplegado en:", address);

  // Guardar ABI
  const abi = JSON.stringify(await artifacts.readArtifact("MainContract"));
  fs.writeFileSync("../frontend/src/contracts/contractABI.json", abi);

  // Guardar dirección
  fs.writeFileSync(
    "../frontend/src/contracts/contract-address.json",
    JSON.stringify({ address })
  );
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
