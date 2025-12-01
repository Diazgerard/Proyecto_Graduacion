import { useContext } from "react";
import { BlockchainContext } from "../context/BlockchainContext";

export default function ConnectWalletButton() {
  const { account, connectWallet } = useContext(BlockchainContext);

  return (
    <button
      onClick={connectWallet}
      className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
    >
      {account ? account.slice(0, 6) + "..." + account.slice(-4) : "Conectar Wallet"}
    </button>
  );
}
