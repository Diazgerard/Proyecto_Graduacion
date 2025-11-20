const { ethers } = require("hardhat");
const fs = require("fs");
const path = require("path");

async function main() {
  console.log("🚀 Iniciando despliegue del contrato MainContract...");

  // Obtener el contrato
  const MainContract = await ethers.getContractFactory("MainContract");
  
  // Desplegar el contrato
  console.log("📦 Desplegando contrato...");
  const mainContract = await MainContract.deploy();
  
  // Esperar confirmación
  await mainContract.waitForDeployment();
  
  const contractAddress = await mainContract.getAddress();
  console.log("✅ MainContract desplegado en:", contractAddress);

  // Guardar la dirección del contrato
  const contractAddressPath = path.join(__dirname, "..", "..", "frontend", "src", "contracts", "contract-address.js");
  const contractAddressContent = `export const CONTRACT_ADDRESS = "${contractAddress}";\n`;
  
  // Crear directorio si no existe
  const contractsDir = path.dirname(contractAddressPath);
  if (!fs.existsSync(contractsDir)) {
    fs.mkdirSync(contractsDir, { recursive: true });
  }
  
  fs.writeFileSync(contractAddressPath, contractAddressContent);
  console.log("📁 Dirección guardada en:", contractAddressPath);

  // Obtener y guardar el ABI
  const artifactPath = path.join(__dirname, "..", "artifacts", "contracts", "MainContract.sol", "MainContract.json");
  const artifact = JSON.parse(fs.readFileSync(artifactPath, "utf8"));
  
  const abiPath = path.join(__dirname, "..", "..", "frontend", "src", "contracts", "contractABI.json");
  const abiContent = JSON.stringify(artifact.abi, null, 2);
  
  fs.writeFileSync(abiPath, abiContent);
  console.log("📁 ABI guardado en:", abiPath);

  console.log("\n🎉 Despliegue completado exitosamente!");
  console.log("📋 Detalles:");
  console.log("   - Contrato: MainContract");
  console.log("   - Dirección:", contractAddress);
  console.log("   - Red:", (await ethers.provider.getNetwork()).name);
  console.log("   - Block:", await ethers.provider.getBlockNumber());
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error("❌ Error en el despliegue:", error);
    process.exit(1);
  });
