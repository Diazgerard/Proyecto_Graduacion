import { configureChains, createClient } from "wagmi";
import { mainnet, goerli } from "wagmi/chains";
import { publicProvider } from "wagmi/providers/public";
import { MetaMaskConnector } from "wagmi/connectors/metaMask";

// Configura las chains y el proveedor
const { chains, provider } = configureChains(
  [mainnet, goerli],
  [publicProvider()]
);

export const wagmiClient = createClient({
  autoConnect: false, // evita conexión automática
  connectors: [
    new MetaMaskConnector({ chains })
  ],
  provider,
});
