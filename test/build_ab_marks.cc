#include "ngfixlib/graph/hnsw_ngfix.h"
#include <chrono>
#include <iostream>
#include <sstream>
using namespace ngfixlib;
int main(int argc,char** argv) {
    try {
        std::string index,gt_path,out,checkpoints; size_t L=200;
        for(int i=1;i<argc;i+=2) {
            if(i+1>=argc) throw std::runtime_error("Expected option value");
            std::string a=argv[i],v=argv[i+1];
            if(a=="--index") index=v; else if(a=="--gt") gt_path=v;
            else if(a=="--out") out=v; else if(a=="--checkpoints") checkpoints=v;
            else if(a=="--L") L=std::stoull(v); else throw std::runtime_error("Unknown option: "+a);
        }
        if(index.empty() || gt_path.empty() || out.empty() || checkpoints.empty() || !L)
            throw std::runtime_error("Require --index --gt (sample-ordered ibin) --out --L --checkpoints CSV");
        if(std::filesystem::exists(out)) throw std::runtime_error("Use a new marks output directory");
        std::vector<uint64_t> sizes; std::stringstream ss(checkpoints); std::string token;
        while(std::getline(ss,token,',')) sizes.push_back(std::stoull(token));
        std::sort(sizes.begin(),sizes.end());
        if(sizes.empty() || !sizes.front() || std::adjacent_find(sizes.begin(),sizes.end())!=sizes.end())
            throw std::runtime_error("Checkpoints must be distinct and positive");
        std::ifstream gt(gt_path,std::ios::binary); uint32_t rows,cols;
        abRead(gt,&rows,4); abRead(gt,&cols,4);
        if(L>cols || sizes.back()>rows || !cols ||
           std::filesystem::file_size(gt_path)<8+uint64_t(rows)*cols*4)
            throw std::runtime_error("GT shape/length incompatible with checkpoint/L");
        validateABIndex(index);
        HNSW_NGFix<float> graph(IP_float,index);
        ABMarks marks=ABMarks::layout(graph); marks.top_l=L;
        if(L>marks.n) throw std::runtime_error("L exceeds base count");
        const size_t edges=marks.offsets.back();
        std::vector<uint32_t> op(edges),ah(edges),bh(edges),excluded(marks.n),top(L);
        std::vector<int32_t> rank(marks.n,-1); uint32_t stamp=0;
        marks.flags.resize(edges);
        std::filesystem::create_directories(out);
        std::ofstream summary(out+"/coverage.csv");
        summary<<"queries,L,total_edges,seen_edges,a_edges,b_edges,cumulative_label_seconds,save_seconds\n";
        using Clock=std::chrono::steady_clock;
        double label_seconds=0; size_t checkpoint=0;
        for(uint64_t q=0;q<sizes.back();++q) {
            abRead(gt,top.data(),L*4); gt.seekg(uint64_t(cols-L)*4,std::ios::cur);
            auto begin=Clock::now(); labelABQuery(graph,top,marks,rank,excluded,stamp,op,ah,bh);
            label_seconds+=std::chrono::duration<double>(Clock::now()-begin).count();
            if((q+1)%1000==0) std::cout<<"Labeled "<<q+1<<" / "<<sizes.back()<<std::endl;
            if(q+1!=sizes[checkpoint]) continue;
            const auto saving=Clock::now(); uint64_t seen=0,a=0,b=0;
            for(size_t e=0;e<edges;++e) {
                if(bh[e]>ah[e] || ah[e]>op[e]) throw std::runtime_error("Label counts violate B subset A");
                marks.flags[e]=(op[e]?1:0)|(ah[e]?2:0)|(bh[e]?4:0);
                seen+=op[e]>0; a+=ah[e]>0; b+=bh[e]>0;
            }
            marks.queries=q+1; marks.save(out+"/q"+std::to_string(q+1)+".abm");
            summary<<q+1<<','<<L<<','<<edges<<','<<seen<<','<<a<<','<<b<<','<<label_seconds<<','
                   <<std::chrono::duration<double>(Clock::now()-saving).count()<<'\n'; summary.flush();
            std::cout<<"Checkpoint "<<q+1<<": A="<<a<<" B="<<b<<" / "<<edges<<std::endl;
            ++checkpoint;
        }
        // Counts at the largest checkpoint support subsequent threshold studies.
        // Smaller checkpoints retain masks, not three extra full count arrays.
        std::ofstream counts(out+"/final_counts.bin",std::ios::binary);
        const uint64_t h[]={marks.n,uint64_t(edges),marks.top_l,marks.queries,marks.fingerprint};
        abWrite(counts,"ABCOUNT1",8); abWrite(counts,h,sizeof(h));
        abWrite(counts,op.data(),edges*4); abWrite(counts,ah.data(),edges*4); abWrite(counts,bh.data(),edges*4);
        counts.flush(); if(!counts || !summary) throw std::runtime_error("Statistics write failed");
        return 0;
    } catch(const std::exception& e) { std::cerr<<"ERROR: "<<e.what()<<'\n'; return 1; }
}
