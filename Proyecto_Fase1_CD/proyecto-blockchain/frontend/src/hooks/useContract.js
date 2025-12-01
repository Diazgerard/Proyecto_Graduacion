import { useContract as useAppKitContract } from "@reown/appkit";
import contractABI from "../contracts/contractABI.json";
import contractAddress from "../contracts/contract-address.json";

export const useContract = () => {
  const { data: contract, write, read } = useAppKitContract({
    address: contractAddress.address,
    abi: contractABI.abi || contractABI,
  });

  const setData = async (data) => {
    if (!contract) return;
    await write("setData", [data]);
  };

  const getData = async () => {
    if (!contract) return "";
    return await read("getData");
  };

  const getTransactions = async () => {
    if (!contract) return [];
    return await read("getTransactions"); // Función que tu contrato debe tener
  };

  return { contract, setData, getData, getTransactions };
};
