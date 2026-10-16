module.exports = {
    uiHost: '127.0.0.1',
    uiPort: Number(process.env.NODE_RED_PORT || 1880),
    httpAdminRoot: '/editor',
    httpNodeRoot: '/',
    flowFile: 'flows.json',
    flowFilePretty: true,
    editorTheme: { projects: { enabled: false } },
    logging: { console: { level: 'info', metrics: false, audit: false } },
    functionExternalModules: false,
    exportGlobalContextKeys: false
};
