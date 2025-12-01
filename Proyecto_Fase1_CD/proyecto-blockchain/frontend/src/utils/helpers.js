export const shortenAddress = (addr) =>
  addr.slice(0, 6) + "..." + addr.slice(-4);

export const formatDate = (timestamp) => new Date(timestamp).toLocaleString();
